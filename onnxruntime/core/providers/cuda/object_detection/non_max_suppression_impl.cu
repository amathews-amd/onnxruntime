/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/* Modifications Copyright (c) Microsoft. */

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include "non_max_suppression_impl.h"
#include "core/providers/cpu/object_detection/non_max_suppression_helper.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"

#include "core/framework/tensor.h"

#include <cub/cub.cuh>
//TODO:fix the warnings
#ifdef _MSC_VER
#pragma warning(disable : 4244)
#endif
namespace onnxruntime {
namespace cuda {

using namespace nms_helpers;

namespace {

struct __align__(16) Box {
  float x1, y1, x2, y2;
};

// This is the width of the bitmask for masking boxes for each thread.
// This needs to be a multiple of 2(a POD width usually) so that division and
// modulo can be implemented as bit operations during host selection.
constexpr int kNmsBoxesPerThread = 8 * sizeof(int);
// Helper to calculate modulo mask and shift bits.
// For kNmsBoxesPerThread=32 ModuloMask will be 31, i.e 0x1F thus
// i % 32 == i & 31. Similarly ShiftBits will be 5 so that
// i / 32 == i >> 5. Using these bit operations should reduce the stall on host
// thread.
__device__ constexpr int NumBits(int n) { return (n == 0) ? 0 : NumBits(n >> 1) + 1; }

constexpr int kNmsBlockDim = 16;
constexpr int kNmsBlockDimMax = 128;

// Check whether two boxes have an IoU greater than threshold.
template <typename T>
__device__ inline bool OverThreshold(const Box* a, const Box* b,
                                     const float a_area,
                                     const T iou_threshold) {
  const float b_area = (b->x2 - b->x1) * (b->y2 - b->y1);
  if (a_area == 0.0f || b_area == 0.0f) return false;
  const float xx1 = fmaxf(a->x1, b->x1);
  const float yy1 = fmaxf(a->y1, b->y1);
  const float xx2 = fminf(a->x2, b->x2);
  const float yy2 = fminf(a->y2, b->y2);

  // fdimf computes the positive difference between xx2+1 and xx1.
  const float w = fdimf(xx2, xx1);
  const float h = fdimf(yy2, yy1);
  const float intersection = w * h;

  // Testing for aa/bb > t
  // eq with aa > bb*t (b is !=0)
  // avoiding divisions.
  const float aa = intersection;
  const float bb = a_area + b_area - intersection;
  const float bt = bb * iou_threshold;
  return aa >= bt;
}

template <typename T>
__device__ inline bool CheckBit(T* bit_mask, int bit) {
  constexpr int kShiftLen = NumBits(8 * sizeof(T)) - 1;
  constexpr int kRemainderMask = 8 * sizeof(T) - 1;
  int bin = bit >> kShiftLen;
  return (bit_mask[bin] >> (bit & kRemainderMask)) & 1;
}

// Produce a global bitmask (result_mask) of selected boxes from bitmask
// generated by NMSKernel Abort early if max_boxes boxes are selected. Bitmask
// is num_boxes*bit_mask_len bits indicating whether to keep or remove a box.
__global__ void NMSReduce(const int* bitmask, const int bit_mask_len,
                          const int num_boxes, const int max_boxes,
                          char* result_mask) {
  extern __shared__ int local[];

  // set global mask to accept all boxes
  for (int box = blockIdx.x * blockDim.x + threadIdx.x; box < bit_mask_len; box += blockDim.x * gridDim.x) {
    local[box] = 0xFFFFFFFF;
  }
  __syncthreads();
  int accepted_boxes = 0;
  for (int box = 0; box < num_boxes - 1; ++box) {
    // if current box is masked by an earlier box, skip it.
    if (!CheckBit(local, box)) {
      continue;
    }
    accepted_boxes += 1;
    int offset = box * bit_mask_len;
    // update global mask with current box's mask
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < bit_mask_len; b += blockDim.x * gridDim.x) {
      local[b] &= ~bitmask[offset + b];
    }
    __syncthreads();
    if (accepted_boxes > max_boxes) break;
  }
  // copy global mask to result_max char array. char array is needed for
  // cub::DeviceSelect later.
  for (int box = blockIdx.x * blockDim.x + threadIdx.x; box < num_boxes; box += blockDim.x * gridDim.x) {
    result_mask[box] = CheckBit(local, box);
  }
}

// For each box, compute a bitmask of boxes which has an overlap with given box
// above threshold.
//
// Starting from highes scoring box, mark any box which has IoU>threshold with
// given box. Each thread processes a kNmsBoxesPerThread boxes per stride, and
// each box has bitmask of overlaps of length bit_mask_len.
//
__launch_bounds__(kNmsBlockDim* kNmsBlockDim, 4) __global__
    void NMSKernel(
        const int64_t center_point_box,
        const Box* d_desc_sorted_boxes,
        const int num_boxes,
        const float iou_threshold,
        const int bit_mask_len,
        int* d_delete_mask) {
  for (int i_block_offset = blockIdx.x * blockDim.x; i_block_offset < num_boxes;
       i_block_offset += blockDim.x * gridDim.x) {
    const int i = i_block_offset + threadIdx.x;
    if (i < num_boxes) {
      for (int j_thread_offset =
               kNmsBoxesPerThread * (blockIdx.y * blockDim.y + threadIdx.y);
           j_thread_offset < num_boxes;
           j_thread_offset += kNmsBoxesPerThread * blockDim.y * gridDim.y) {
        // Note : We can do everything using multiplication,
        // and use fp16 - we are comparing against a low precision
        // threshold.
        int above_threshold = 0;
        // Make sure that threads are within valid domain.
        bool valid = false;
        // Loop over the next kNmsBoxesPerThread boxes and set corresponding bit
        // if it is overlapping with current box
        for (int ib = 0; ib < kNmsBoxesPerThread; ++ib) {
          // This thread will compare Box i and Box j.
          const int j = j_thread_offset + ib;
          if (i >= j || i >= num_boxes || j >= num_boxes) continue;
          valid = true;
          if (SuppressByIOU(reinterpret_cast<const float*>(d_desc_sorted_boxes),
                            i, j, center_point_box, iou_threshold)) {
            // we have score[j] <= score[i].
            above_threshold |= (1U << ib);
          }
        }
        if (valid) {
          d_delete_mask[i * bit_mask_len + j_thread_offset / kNmsBoxesPerThread] =
              above_threshold;
        }
      }
    }
  }
}
// Variadic template helpers for Index selecting multiple arrays at the same
// time
template <typename Index>
__device__ inline void SelectHelper(const Index i_selected,
                                    const Index i_original) {}

template <typename Index, typename T, typename... Args>
__device__ inline void SelectHelper(const Index i_selected,
                                    const Index i_original,
                                    const T* original, T* selected,
                                    Args... args) {
  selected[i_selected] = original[i_original];
  SelectHelper(i_selected, i_original, args...);
}

// Helper template to select elements from original arrays using the index
// mapping and store into selected array. Each array sharing same mapping need
// to be passed as pairs of pointers to original and selected arrays. For
// selecting 2 arrays call would be
// IndexMultiSelect(num_elements, indices, original1 ,selected1, original2,
// selected2).
template <typename Index, typename T, typename... Args>
__global__ void IndexMultiSelect(const int num_elements, const Index* indices,
                                 const T* original, T* selected, Args... args) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements; idx += blockDim.x * gridDim.x) {
    SelectHelper(idx, indices[idx], original, selected, args...);
  }
}

template <typename T>
__global__ void SetZero(const int count, T* __restrict__ ptr) {
  // Check that the grid is one dimensional and index doesn't overflow.
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  assert(blockDim.x * gridDim.x / blockDim.x == gridDim.x);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    ptr[i] = T(0);
  }
}

template <typename T>
__global__ void Iota(const int num_elements, const T offset, T* to_fill) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements; idx += blockDim.x * gridDim.x) {
    to_fill[idx] = static_cast<T>(idx) + offset;
  }
}

__global__ void NormalizeOutput(const int num_elements, const int* original, int64_t* to_normalize, int64_t batch_index, int64_t class_index) {
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements; idx += blockDim.x * gridDim.x) {
    to_normalize[idx * 3] = batch_index;
    to_normalize[idx * 3 + 1] = class_index;
    to_normalize[idx * 3 + 2] = static_cast<int64_t>(original[idx]);
  }
}

Status NmsGpu(cudaStream_t stream,
              std::function<IAllocatorUniquePtr<void>(size_t)> allocator,
              const int64_t center_point_box,
              const float* d_sorted_boxes_float_ptr,
              const int num_boxes,
              const float iou_threshold,
              int* d_selected_indices,
              int* h_nkeep,
              const int max_boxes) {
  // Making sure we respect the __align(16)__
  // we promised to the compiler.
  auto iptr = reinterpret_cast<std::uintptr_t>(d_sorted_boxes_float_ptr);
  ORT_ENFORCE((iptr & 15) == 0);

  const int bit_mask_len =
      (num_boxes + kNmsBoxesPerThread - 1) / kNmsBoxesPerThread;
  int max_nms_mask_size = num_boxes * bit_mask_len;

  IAllocatorUniquePtr<void> d_nms_mask_ptr{allocator(max_nms_mask_size * sizeof(int))};
  auto* d_nms_mask = static_cast<int*>(d_nms_mask_ptr.get());

  int blocksPerGrid = (int)(ceil(static_cast<float>(max_nms_mask_size) / GridDim::maxThreadsPerBlock));
  SetZero<int><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(max_nms_mask_size, d_nms_mask);

  int* d_delete_mask = d_nms_mask;
  int* h_selected_count = h_nkeep;
  const Box* d_sorted_boxes =
      reinterpret_cast<const Box*>(d_sorted_boxes_float_ptr);
  dim3 block_dim, thread_block;
  int num_blocks = (num_boxes + kNmsBlockDim - 1) / kNmsBlockDim;
  num_blocks = std::max(std::min(num_blocks, kNmsBlockDimMax), 1);
  block_dim.x = num_blocks;
  block_dim.y = num_blocks;
  block_dim.z = 1;
  thread_block.x = kNmsBlockDim;
  thread_block.y = kNmsBlockDim;
  thread_block.z = 1;
  NMSKernel<<<block_dim, thread_block, 0, stream>>>(center_point_box,
                                         d_sorted_boxes,
                                         num_boxes,
                                         iou_threshold,
                                         bit_mask_len,
                                         d_delete_mask);

  IAllocatorUniquePtr<void> d_selected_boxes_ptr{allocator(num_boxes * sizeof(char))};
  auto* d_selected_boxes = static_cast<char*>(d_selected_boxes_ptr.get());
  IAllocatorUniquePtr<void> d_indices_ptr{allocator(num_boxes * sizeof(int))};
  auto* d_indices = static_cast<int*>(d_indices_ptr.get());

  blocksPerGrid = (int)(ceil(static_cast<float>(num_boxes) / GridDim::maxThreadsPerBlock));
  Iota<int><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(num_boxes, 0, d_indices);

  NMSReduce<<<1, 1024, bit_mask_len * sizeof(int), stream>>>(d_delete_mask, bit_mask_len, num_boxes, max_boxes, d_selected_boxes);

  size_t flagged_buffer_size = 0;
  CUDA_RETURN_IF_ERROR(cub::DeviceSelect::Flagged(static_cast<void*>(nullptr),  // temp_storage
                                                  flagged_buffer_size,
                                                  static_cast<int*>(nullptr),   // input
                                                  static_cast<char*>(nullptr),  // selection flag
                                                  static_cast<int*>(nullptr),   // selected items
                                                  static_cast<int*>(nullptr),   // num_selected
                                                  num_boxes,
                                                  stream));

  IAllocatorUniquePtr<void> d_cub_scratch_buffer_ptr{allocator(flagged_buffer_size)};
  auto* d_cub_scratch_buffer = static_cast<uint8_t*>(d_cub_scratch_buffer_ptr.get());
  IAllocatorUniquePtr<void> d_num_selected_ptr{allocator(sizeof(int))};
  auto* d_num_selected = static_cast<int*>(d_num_selected_ptr.get());

  CUDA_RETURN_IF_ERROR(cub::DeviceSelect::Flagged(
      d_cub_scratch_buffer,  // temp_storage
      flagged_buffer_size,
      d_indices,           // input
      d_selected_boxes,    // selection flag
      d_selected_indices,  // selected items
      d_num_selected, num_boxes, stream));
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(h_selected_count, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost, stream));
  // cudaStreamSynchronize is needed since the value of h_selected_count will be used by host after this function.
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

  return Status::OK();
}

struct DeviceGreaterThan {
  float threshold_;
  __host__ __device__ __forceinline__ DeviceGreaterThan(float threshold)
      : threshold_(threshold) {}

  __host__ __device__ __forceinline__ bool operator()(const float& val) const {
    return (val > threshold_);
  }
};

}  // namespace

Status NonMaxSuppressionImpl(
    cudaStream_t stream,
    std::function<IAllocatorUniquePtr<void>(size_t)> allocator,
    const PrepareContext& pc,
    const int64_t center_point_box,
    int64_t batch_index,
    int64_t class_index,
    int max_output_boxes_per_class,
    float iou_threshold,
    float score_threshold,
    IAllocatorUniquePtr<void>& selected_indices,
    int* h_number_selected) {
  // STEP 1. Prepare data
  int num_boxes = pc.num_boxes_;
  const float* boxes_data = pc.boxes_data_ + batch_index * num_boxes * 4;
  const float* scores_data = pc.scores_data_ + (batch_index * pc.num_classes_ + class_index) * num_boxes;

  // prepare temporary memory for sorting scores

  // calculate temporary size that used for sorting
  size_t cub_sort_temp_storage_bytes = 0;
  CUDA_RETURN_IF_ERROR(cub::DeviceRadixSort::SortPairsDescending(
      nullptr, cub_sort_temp_storage_bytes,
      static_cast<float*>(nullptr),  // scores
      static_cast<float*>(nullptr),  // sorted scores
      static_cast<int*>(nullptr),    // input indices
      static_cast<int*>(nullptr),    // sorted indices
      num_boxes,                     // num items
      0, 8 * sizeof(float),           // sort all bits
      stream));

  // allocate temporary memory
  IAllocatorUniquePtr<void> d_cub_sort_buffer_ptr{allocator(cub_sort_temp_storage_bytes)};
  auto* d_cub_sort_buffer = static_cast<uint8_t*>(d_cub_sort_buffer_ptr.get());
  IAllocatorUniquePtr<void> d_indices_ptr{allocator(num_boxes * sizeof(int))};
  auto* d_indices = static_cast<int*>(d_indices_ptr.get());
  IAllocatorUniquePtr<void> d_sorted_indices_ptr{allocator(num_boxes * sizeof(int))};
  auto* d_sorted_indices = static_cast<int*>(d_sorted_indices_ptr.get());
  IAllocatorUniquePtr<void> d_selected_indices_ptr{allocator(num_boxes * sizeof(int))};
  auto* d_selected_indices = static_cast<int*>(d_selected_indices_ptr.get());
  IAllocatorUniquePtr<void> d_sorted_scores_ptr{allocator(num_boxes * sizeof(float))};
  auto* d_sorted_scores = static_cast<float*>(d_sorted_scores_ptr.get());
  IAllocatorUniquePtr<void> d_sorted_boxes_ptr{allocator(num_boxes * 4 * sizeof(float))};
  auto* d_sorted_boxes = static_cast<float*>(d_sorted_boxes_ptr.get());

  // create sequense of indices
  int blocksPerGrid = (int)(ceil(static_cast<float>(num_boxes) / GridDim::maxThreadsPerBlock));
  Iota<int><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(num_boxes, 0, d_indices);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  // sort scores
  CUDA_RETURN_IF_ERROR(cub::DeviceRadixSort::SortPairsDescending(
      d_cub_sort_buffer,
      cub_sort_temp_storage_bytes,
      scores_data,
      d_sorted_scores,
      d_indices,
      d_sorted_indices,
      num_boxes,
      0,
      8 * sizeof(float),  // sort all bits
      stream));

  // pick sorted scores
  const Box* original_boxes = reinterpret_cast<const Box*>(boxes_data);
  Box* sorted_boxes = reinterpret_cast<Box*>(d_sorted_boxes);
  IndexMultiSelect<int, Box><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(num_boxes, d_sorted_indices, original_boxes, sorted_boxes);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  // STEP 2. filter boxes by scores
  int limited_num_boxes = num_boxes;
  if (pc.score_threshold_ != nullptr) {
    thrust::device_ptr<float> sorted_scores_device_ptr(d_sorted_scores);
    limited_num_boxes = thrust::count_if(
        thrust::cuda::par.on(stream),
        sorted_scores_device_ptr,
        sorted_scores_device_ptr + num_boxes,
        DeviceGreaterThan(score_threshold));
    CUDA_RETURN_IF_ERROR(cudaGetLastError());

    if (limited_num_boxes == 0) {
      *h_number_selected = 0;
      return Status::OK();
    }
  }

  // STEP 3. launch NMS kernels
  ORT_RETURN_IF_ERROR(NmsGpu(stream,
                             allocator,
                             center_point_box,
                             d_sorted_boxes,
                             limited_num_boxes,
                             iou_threshold,
                             d_selected_indices,
                             h_number_selected,
                             max_output_boxes_per_class));
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  // STEP 4. map back to sorted indices
  *h_number_selected = std::min(*h_number_selected, max_output_boxes_per_class);
  int num_to_keep = *h_number_selected;
  if (num_to_keep > 0) {
    IAllocatorUniquePtr<void> d_output_indices_ptr{allocator(num_to_keep * sizeof(int))};
    auto* d_output_indices = static_cast<int*>(d_output_indices_ptr.get());
    IAllocatorUniquePtr<void> d_normalized_output_indices_ptr{allocator(num_to_keep * 3 * sizeof(int64_t))};
    auto* d_normalized_output_indices = static_cast<int64_t*>(d_normalized_output_indices_ptr.get());

    blocksPerGrid = (int)(ceil(static_cast<float>(num_to_keep) / GridDim::maxThreadsPerBlock));
    IndexMultiSelect<int, int><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(num_to_keep, d_selected_indices, d_sorted_indices, d_output_indices);
    NormalizeOutput<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(num_to_keep, d_output_indices, d_normalized_output_indices, batch_index, class_index);
    CUDA_RETURN_IF_ERROR(cudaGetLastError());

    selected_indices = std::move(d_normalized_output_indices_ptr);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
