//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of
// Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#include "ops/commuting.hpp"
#include "common/parallel_for.hpp"
#include "storage/all_storage.hpp"

#define DEVICE_SWITCH(cpu, gpu, a, b, out)                                     \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, b, out);                                                            \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, b, out);                                                            \
  }

#define ADD_KERNEL()                                                           \
  const tcapint I_a = a.stride[0U];                                            \
  const tcapint I_b = b.stride[0U];                                            \
  const tcapint I_o = out.stride[0U];                                          \
  const size_t n = out.get_size();                                             \
  pfControl.par_for(0, n, [&](const tcapint &i, const unsigned &cpu) {         \
    po[i * I_o] = pa[i * I_a] + pb[i * I_b];                                   \
  })

#define MUL_KERNEL()                                                           \
  const tcapint I_a = a.stride[0U];                                            \
  const tcapint I_b = b.stride[0U];                                            \
  const tcapint I_o = out.stride[0U];                                          \
  const size_t n = out.get_size();                                             \
  pfControl.par_for(0, n, [&](const tcapint &i, const unsigned &cpu) {         \
    po[i * I_o] = pa[i * I_a] * pb[i * I_b];                                   \
  })

#define DISPATCH_GPU_KERNEL(type, type2, api_call)                             \
  const tcapint args[10U]{                                                     \
      a.offset,       a.stride[0U], b.offset, b.stride[0U], out.offset,        \
      out.stride[0U], 0U,           0U,       0U,           0U};               \
  std::shared_ptr<type> a_storage =                                            \
      std::dynamic_pointer_cast<type>(a.storage);                              \
  std::shared_ptr<type2> b_storage =                                           \
      std::dynamic_pointer_cast<type2>(b.storage);                             \
  std::shared_ptr<type> o_storage =                                            \
      std::dynamic_pointer_cast<type>(out.storage);                            \
  a_storage->dev->RequestKernel(                                               \
      api_call, args, out.get_size(),                                          \
      {a_storage->buffer, b_storage->buffer, o_storage->buffer})

namespace Weed {
static void cpu_real_add(const Tensor &a, const Tensor &b, Tensor &out) {
  CAST_STORAGE(pa, a, real1, CpuRealStorage);
  CAST_STORAGE(pb, b, real1, CpuRealStorage);
  CAST_STORAGE(po, out, real1, CpuRealStorage);

  ADD_KERNEL();
}
static void cpu_complex_add(const Tensor &a, const Tensor &b, Tensor &out) {
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(pb, b, complex, CpuComplexStorage);
  CAST_STORAGE(po, out, complex, CpuComplexStorage);

  ADD_KERNEL();
}
static void cpu_mixed_add(const Tensor &a, const Tensor &b, Tensor &out) {
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(pb, b, real1, CpuRealStorage);
  CAST_STORAGE(po, out, complex, CpuComplexStorage);

  ADD_KERNEL();
}

#if ENABLE_GPU
static void gpu_real_add(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, GpuRealStorage, OCL_API_ADD_REAL);
}
static void gpu_complex_add(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuComplexStorage,
                      OCL_API_ADD_COMPLEX);
}
static void gpu_mixed_add(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuRealStorage, OCL_API_ADD_MIXED);
}
#endif

static void cpu_real_mul(const Tensor &a, const Tensor &b, Tensor &out) {
  CAST_STORAGE(pa, a, real1, CpuRealStorage);
  CAST_STORAGE(pb, b, real1, CpuRealStorage);
  CAST_STORAGE(po, out, real1, CpuRealStorage);

  MUL_KERNEL();
}
static void cpu_complex_mul(const Tensor &a, const Tensor &b, Tensor &out) {
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(pb, b, complex, CpuComplexStorage);
  CAST_STORAGE(po, out, complex, CpuComplexStorage);

  MUL_KERNEL();
}
static void cpu_mixed_mul(const Tensor &a, const Tensor &b, Tensor &out) {
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(pb, b, real1, CpuRealStorage);
  CAST_STORAGE(po, out, complex, CpuComplexStorage);

  MUL_KERNEL();
}

#if ENABLE_GPU
static void gpu_real_mul(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, GpuRealStorage, OCL_API_MUL_REAL);
}
static void gpu_complex_mul(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuComplexStorage,
                      OCL_API_MUL_COMPLEX);
}
static void gpu_mixed_mul(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuRealStorage, OCL_API_MUL_MIXED);
}
#endif

void CommutingKernel::commuting(const Tensor &a, const Tensor &b, Tensor &out) {
  const bool isAComplex = a.storage->dtype == DType::COMPLEX;
  const bool isBComplex = b.storage->dtype == DType::COMPLEX;
  const bool isOutComplex = out.storage->dtype == DType::COMPLEX;
  if (!isOutComplex && (isAComplex || isBComplex)) {
    throw std::invalid_argument(
        "Cannot combine complex tensors into real1 tensor!");
  }
  if (isOutComplex && (!isAComplex && !isBComplex)) {
    throw std::invalid_argument("Output tensor dtype mismatch!");
  }
  const tcapint aSize = a.get_broadcast_size();
  const tcapint bSize = b.get_broadcast_size();
  const tcapint outSize = out.get_broadcast_size();
  if (aSize != bSize) {
    throw std::invalid_argument(
        "In Weed::commuting(a, b, out), 'a' size does not match 'b' size!");
  }
  if (aSize != outSize) {
    throw std::invalid_argument(
        "In Weed::commuting(a, b, out), out size does not match input size!");
  }
  if (isAComplex && isBComplex) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_complex, gpu_complex, a, b, out);
#else
    cpu_complex(a, b, out);
#endif
  } else if (isAComplex) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_mixed, gpu_mixed, a, b, out);
#else
    cpu_mixed(a, b, out);
#endif
  } else if (isBComplex) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_mixed, gpu_mixed, b, a, out);
#else
    cpu_mixed(b, a, out);
#endif
  } else {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_real, gpu_real, a, b, out);
#else
    cpu_real(a, b, out);
#endif
  }
}

CommutingKernel add_kernel = {cpu_real_add,  cpu_complex_add,
                              cpu_mixed_add,
#if ENABLE_GPU
                              gpu_real_add,  gpu_complex_add,
                              gpu_mixed_add
#endif
};

CommutingKernel mul_kernel = {cpu_real_mul,  cpu_complex_mul,
                              cpu_mixed_mul,
#if ENABLE_GPU
                              gpu_real_mul,  gpu_complex_mul,
                              gpu_mixed_mul
#endif
};

void add(const Tensor &a, const Tensor &b, Tensor &out) {
  add_kernel.commuting(a, b, out);
}
void mul(const Tensor &a, const Tensor &b, Tensor &out) {
  mul_kernel.commuting(a, b, out);
}
} // namespace Weed
