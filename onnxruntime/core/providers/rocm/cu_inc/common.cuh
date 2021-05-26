// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include <vector>
#include <mutex>
#include <assert.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/shared_inc/rocm_call.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
__device__ __inline__ T _Ceil(T a);

template <>
__device__ __inline__ float _Ceil(float a) { return ceilf(a); }

template <>
__device__ __inline__ double _Ceil(double a) { return ceil(a); }

template <>
__device__ __inline__ half _Ceil(half a) { return half(ceilf((float)a)); }

template <typename T>
__device__ __inline__ T _Floor(T a);

template <>
__device__ __inline__ float _Floor(float a) { return floorf(a); }

template <>
__device__ __inline__ double _Floor(double a) { return floor(a); }

template <>
__device__ __inline__ half _Floor(half a) { return half(floorf((float)a)); }

template <typename T>
__device__ __inline__ T _Sqrt(T a);

template <>
__device__ __inline__ float _Sqrt(float a) { return sqrtf(a); }

template <>
__device__ __inline__ double _Sqrt(double a) { return sqrt(a); }

template <>
__device__ __inline__ half _Sqrt(half a) { return half(sqrtf((float)a)); }

template <typename T>
__device__ __inline__ T _Erf(T a);

template <>
__device__ __inline__ float _Erf(float a) { return erff(a); }

template <>
__device__ __inline__ double _Erf(double a) { return erf(a); }

template <>
__device__ __inline__ half _Erf(half a) { return half(erff((float)a)); }

template <typename T>
__device__ __inline__ T _Round(T a);

template <>
__device__ __inline__ float _Round(float a) { return rintf(a); }

template <>
__device__ __inline__ double _Round(double a) { return rint(a); }

template <>
__device__ __inline__ half _Round(half a) { 
  return hrint(a);
}

template <typename T>
__device__ __inline__ T _Cos(T a);

template <>
__device__ __inline__ float _Cos(float a) { return cosf(a); }

template <>
__device__ __inline__ double _Cos(double a) { return cos(a); }

template <>
__device__ __inline__ half _Cos(half a) { return hcos(a); }

template <typename T>
__device__ __inline__ T _Sin(T a);

template <>
__device__ __inline__ float _Sin(float a) { return sinf(a); }

template <>
__device__ __inline__ double _Sin(double a) { return sin(a); }

template <>
__device__ __inline__ half _Sin(half a) { return hsin(a); }

template <typename T>
__device__ __inline__ T _Exp(T a);

template <>
__device__ __inline__ float _Exp(float a) { return expf(a); }

template <>
__device__ __inline__ double _Exp(double a) { return exp(a); }

template <>
__device__ __inline__ half _Exp(half a) { return half(expf((float)a)); }

template <typename T>
__device__ __inline__ T _Log(T a);

template <>
__device__ __inline__ float _Log(float a) { return logf(a); }

template <>
__device__ __inline__ double _Log(double a) { return log(a); }

template <>
__device__ __inline__ half _Log(half a) { return half(logf((float)a)); }

template <typename T>
__device__ __inline T _Tanh(T a);

template <>
__device__ __inline__ float _Tanh(float a) { return tanhf(a); }

template <>
__device__ __inline__ double _Tanh(double a) { return tanh(a); }

template <>
__device__ __inline__ half _Tanh(half a) { return half(tanhf((float)a)); }

template <>
__device__ __inline__ half2 _Tanh(half2 a) {
  float2 tmp = (__half22float2(a));
  tmp.x = tanhf(tmp.x);
  tmp.y = tanhf(tmp.y);
  return __float22half2_rn(tmp);
}

// Capture permutations of int32/64/float/double
template <typename T, typename T1>
__device__ __inline__ T _Pow(T a, T1 b) {
  return static_cast<T>(pow(static_cast<double>(a), static_cast<double>(b)));
}

template <>
__device__ __inline__ float _Pow(float a, float b) { return powf(a, b); }

template <>
__device__ __inline__ double _Pow(double a, double b) { return pow(a, b); }

template <>
__device__ __inline__ half _Pow(half a, half b) { return half(powf((float)a, (float)b)); }

template <typename T>
__device__ __inline__ T _Min(T a, T b) { return a < b ? a : b; }

template <typename T>
__device__ __inline__ T _Max(T a, T b) { return a > b ? a : b; }

template <typename T>
__device__ __inline__ T _Abs(T a) { return a > (T)0 ? a : -a; }

/////////////////////////////////////
#define BUILTIN_RCP_F32 __builtin_amdgcn_rcpf
#define MATH_FAST_RCP(X) BUILTIN_RCP_F32(X)
#define BUILTIN_FMA_F32 __builtin_fmaf
#define BUILTIN_MAD_F32 __ocml_fma_f32
//__ocml_fmuladd_f32
#define MATH_MAD(A,B,C) BUILTIN_MAD_F32(A, B, C)
#define BUILTIN_ABS_F32 __builtin_fabsf
#define AS_FLOAT(X) __builtin_astype(X, float)
#define PINFBITPATT_SP32  0x7f800000
#define BUILTIN_COPYSIGN_F32 __builtin_copysignf


__device__ __inline__ float erfcx_new_private(float x)
{
    float n = x - 2.0f;
    float d = x + 2.0f;
    float r = MATH_FAST_RCP(d);
    float q = n * r;
    float e = BUILTIN_FMA_F32(-q, x, BUILTIN_FMA_F32(q + 1.0f, -2.0f, x));
    q = BUILTIN_FMA_F32(r, e, q);

    float p = MATH_MAD(q, MATH_MAD(q, MATH_MAD(q, MATH_MAD(q,
              MATH_MAD(q, MATH_MAD(q, MATH_MAD(q, MATH_MAD(q,
              MATH_MAD(q,
                  -0x1.adf188p-12f, -0x1.45aea6p-10f),
                  0x1.5a5f68p-10f), 0x1.1b44cep-7f),
                  -0x1.082b62p-7f), -0x1.bc143p-5f),
                  0x1.4ffc54p-3f), -0x1.5407fap-3f),
                  -0x1.7bf616p-4f), 0x1.1ba038p-2);
    float tx = x + x;
    d = 1.0f + tx;
    r = MATH_FAST_RCP(d);
    q = BUILTIN_FMA_F32(p, r, r);
    e = BUILTIN_FMA_F32(-q, tx, 1.0f) + (p - q);
    q = BUILTIN_FMA_F32(r, e, q);
    return q;
}

__device__ __inline__ float erfcx_new(float x)
{
    float ax = BUILTIN_ABS_F32(x);
    float ret;

    if (ax < 0x1.41bbf8p+3f) {
        ret =  erfcx_new_private(ax); //MATH_PRIVATE(erfcx)(ax);
    } else {
        float r = MATH_FAST_RCP(0x1.0p-2f * ax);
        float t = r*r * 0x1.0p-4f;
        float p = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                      6.5625f, -1.875f), 0.75f), -0.5f), 1.0f);
        ret = 0x1.20dd76p-3f * r * p;
    }

    if (x < 0.0f) {
        float x2h = x*x;
        float x2l = BUILTIN_FMA_F32(x, x, -x2h);
        float e = __ocml_exp_f32(x2h); //MATH_MANGLE(exp)(x2h);
        ret = BUILTIN_FMA_F32(2.0f, BUILTIN_FMA_F32(e, x2l, e), -ret);
        ret = x < -0x1.2d6abcp+3f ? INFINITY : ret; // AS_FLOAT(PINFBITPATT_SP32) : ret;
    }

    return ret;
}

__device__ __inline__ float erfc_new(float x)
{
    float ax = BUILTIN_ABS_F32(x);
    float x2h = -x*x;
    float x2l = BUILTIN_FMA_F32(-x, x, -x2h);
    float e = __ocml_exp_f32(x2h); //MATH_MANGLE(exp)(x2h);
    e = BUILTIN_FMA_F32(e, x2l, e);
    float ret = e * erfcx_new(ax); //MATH_PRIVATE(erfcx)(ax);
    ret = ax > 0x1.41bbf8p+3f ? 0.0f : ret;
    float nret = 2.0f - ret;
    return x < 0.0f ? nret : ret;
}


__device__ __inline__ float normcdff_2ae92e(float x)
{
    const float chi = -0x1.6a09e6p-1f;
    const float clo = -0x1.9fcef4p-27f;
    const float b = 0x1.c57228p+3f;
    x = BUILTIN_ABS_F32(x) > b ? BUILTIN_COPYSIGN_F32(b, x) : x;
    float thi = chi * x;
    float tlo = BUILTIN_FMA_F32(clo, x, BUILTIN_FMA_F32(chi, x, -thi));
    float yhi = thi + tlo;
    float ylo = tlo - (yhi - thi);
    float r = erfc_new(yhi); //MATH_MANGLE(erfc)(yhi);
    float dr = -2.0f * yhi * r;
    dr = x >= -1.0f ? 0.0f : dr;
    r = BUILTIN_FMA_F32(ylo, dr, r);
    return 0.5f * r;
}

__device__ __inline__ float normcdff_resimplified(float x)
{
    float x2 = x*x;
    float p = x*BUILTIN_FMA_F32(x2, BUILTIN_FMA_F32(x2, 0x1.cbea4cp-13f, -0x1.29e076p-4f), -0x1.988424p+0f);
    return 1.0f / (1.0f + __ocml_exp_f32(p));
}


#undef BUILTIN_RCP_F32
#undef MATH_FAST_RCP
#undef BUILTIN_FMA_F32
#undef BUILTIN_MAD_F32
#undef MATH_MAD
#undef BUILTIN_ABS_F32
#undef AS_FLOAT
#undef PINFBITPATT_SP32
#undef BUILTIN_COPYSIGN_F32
////////////////////////////////////



template <typename T>
__device__ __inline__ T _Normcdf(T a);

template <>
__device__ __inline__ float _Normcdf(float a) { return normcdff_2ae92e(a); }

template <>
__device__ __inline__ double _Normcdf(double a) { return normcdf(a); }

template <>
__device__ __inline__ half _Normcdf(half a) { return half(normcdff_2ae92e((float)a)); }

template <typename T>
__device__ __inline__ T _Gelu(T a) {
  return a * _Normcdf(a);
}


// We would like to use 64-bit integer to support large matrices. However, HIP seems to support only 32-bit integer
// For now, use int32_t to ensure that both Linux and Windows see this as 32 bit integer type.
#ifndef HIP_LONG
#define HIP_LONG int32_t
#endif

template <class INT, class INT2>
inline __host__ __device__ INT CeilDiv(INT a, INT2 b)  // ceil(a/b)
{
  return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);  // these size_t casts are necessary since b may be INT_MAX (for maxGridSize[])
}

struct GridDim {
  enum : HIP_LONG {
    maxThreadsPerBlock = 256,  // max threads per block
    maxElementsPerThread = 4,  // max element processed per thread
  };
};


#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N)          \
  HIP_LONG id = blockDim.x * blockIdx.x + threadIdx.x;     \
  if (id >= N)                                              \
    return;

#define HIP_KERNEL_ASSERT(...)

// WARP related definitions and functions
constexpr int GPU_WARP_SIZE = 64;

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
  return __shfl(value, srcLane, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
  return __shfl_xor(value, laneMask, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_UP(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
  return __shfl_up(value, delta, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = GPU_WARP_SIZE, unsigned int mask = 0xffffffff)
{
  return __shfl_down(value, delta, width);
}

}  // namespace rocm
}  // namespace onnxruntime
