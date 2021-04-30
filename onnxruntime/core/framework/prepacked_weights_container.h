// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/buffer_deleter.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/allocator.h"
#include "core/platform/ort_mutex.h"
#include <unordered_map>
#include <string>

namespace onnxruntime {

struct PrepackedWeight final {
  // Some weights may be associated with multiple pre-packed buffers.
  // Hence we hold them in containers. It is upto the developer implementing each PrePack()
  // method to define what gets stored in which position of the containers.

  // NOTE: Not all fields may be filled in and not all containers will have the same number of elements
  // It is upto the developer of the kernel to decide which fields to cache for re-use.

  std::vector<std::unique_ptr<void, BufferDeleter>> buffers_;  // cache pre-packed buffers associated with the kernel
  std::vector<size_t> weights_sizes_;                          // cache sizes associated with pre-packed buffers
  std::vector<TensorShape> shapes_;                            // cache tensor shapes associates with pre-packed buffers
  std::vector<bool> flags_;                                    // cache some flags associated with the pre-packed buffers

  bool is_filled_ = false;  // By default, an instance of this class is "unfilled"
};

class PrepackedWeightsContainer final {
 public:
  PrepackedWeightsContainer() {
  }

  ~PrepackedWeightsContainer() = default;

  AllocatorPtr GetAllocator(const std::string& device_name);

  const PrepackedWeight& GetCachedWeight(const std::string& key);

  void WriteCachedWeight(const std::string& key, PrepackedWeight&& packed_weight);

  bool HasCachedWeight(const std::string& key);

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PrepackedWeightsContainer);

  // Resource to be acquired by the method that is going to invoke calls to the kernels'
  // PrePack() methods and does the read/write into the pre-packed weights' container.
  // We only want to invoke PrePack() on a kernel that doesn't have a cached version
  // of its pre-packed weight.
  OrtMutex mutex_;

 private:
  // Define allocators ahead of the container containing tensors because the allocators
  // needs to destructed after the container containing the pre-packed cached tensors
  // because the Tensor buffers will be de-allocated using these allocators
  std::unordered_map<std::string, AllocatorPtr> allocators_;
  std::unordered_map<std::string, PrepackedWeight> initialized_tensor_name_to_prepacked_weights_;
};

}  // namespace onnxruntime
