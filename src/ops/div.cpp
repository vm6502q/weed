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

#include "ops/div.hpp"
#include "common/parallel_for.hpp"
#include "storage/all_storage.hpp"

#define DIV_KERNEL()                                                           \
  const vecCapIntGpu I_a = (vecCapIntGpu)a.stride[0U];                         \
  const vecCapIntGpu I_b = (vecCapIntGpu)b.stride[0U];                         \
  const vecCapIntGpu I_o = (vecCapIntGpu)out.stride[0U];                       \
  const size_t n = out.get_size();                                             \
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {    \
    po[i * I_o] = pa[i * I_a] / pb[i * I_b];                                   \
  })

#define DISPATCH_GPU_KERNEL(type, type2, type3, api_call)                      \
  const vecCapIntGpu args[10U]{(vecCapIntGpu)(a.offset),                       \
                               (vecCapIntGpu)(a.stride[0U]),                   \
                               (vecCapIntGpu)(b.offset),                       \
                               (vecCapIntGpu)(b.stride[0U]),                   \
                               (vecCapIntGpu)(out.offset),                     \
                               (vecCapIntGpu)(out.stride[0U]),                 \
                               0U,                                             \
                               0U,                                             \
                               0U,                                             \
                               0U};                                            \
  std::shared_ptr<type> a_storage =                                            \
      std::dynamic_pointer_cast<type>(a.storage);                              \
  std::shared_ptr<type2> b_storage =                                           \
      std::dynamic_pointer_cast<type2>(b.storage);                             \
  std::shared_ptr<type3> o_storage =                                           \
      std::dynamic_pointer_cast<type3>(out.storage);                           \
  a_storage->dev->RequestKernel(                                               \
      api_call, args, out.get_size(),                                          \
      {a_storage->buffer, b_storage->buffer, o_storage->buffer})

#define DEVICE_SWITCH(cpu, gpu)                                                \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, b, out);                                                            \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, b, out);                                                            \
  }

namespace Weed {
void DivKernel::cpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
  CAST_STORAGE(pa, a, real1, CpuRealStorage);
  CAST_STORAGE(pb, b, real1, CpuRealStorage);
  CAST_STORAGE(po, out, real1, CpuRealStorage);

  DIV_KERNEL();
}
void DivKernel::cpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(pb, b, complex, CpuComplexStorage);
  CAST_STORAGE(po, out, complex, CpuComplexStorage);

  DIV_KERNEL();
}
void DivKernel::cpu_mixed_c_left(const Tensor &a, const Tensor &b,
                                 Tensor &out) {
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(pb, b, real1, CpuRealStorage);
  CAST_STORAGE(po, out, complex, CpuComplexStorage);

  DIV_KERNEL();
}
void DivKernel::cpu_mixed_c_right(const Tensor &a, const Tensor &b,
                                  Tensor &out) {
  CAST_STORAGE(pa, a, real1, CpuRealStorage);
  CAST_STORAGE(pb, b, complex, CpuComplexStorage);
  CAST_STORAGE(po, out, complex, CpuComplexStorage);

  DIV_KERNEL();
}

#if ENABLE_GPU
void DivKernel::gpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, GpuRealStorage, GpuRealStorage,
                      OCL_API_DIV_REAL);
}
void DivKernel::gpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuComplexStorage, GpuComplexStorage,
                      OCL_API_DIV_COMPLEX);
}
void DivKernel::gpu_mixed_c_left(const Tensor &a, const Tensor &b,
                                 Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuRealStorage, GpuComplexStorage,
                      OCL_API_DIV_MIXED_C_LEFT);
}
void DivKernel::gpu_mixed_c_right(const Tensor &a, const Tensor &b,
                                  Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, GpuComplexStorage, GpuComplexStorage,
                      OCL_API_DIV_MIXED_C_RIGHT);
}
#endif

void DivKernel::div(const Tensor &a, const Tensor &b, Tensor &out) {
  const size_t aSize = a.get_broadcast_size();
  const size_t bSize = b.get_broadcast_size();
  const size_t outSize = out.get_broadcast_size();
  if (aSize != bSize) {
    throw std::invalid_argument(
        "In Weed::div(a, b, out), 'a' size does not match 'b' size!");
  }
  if (aSize != outSize) {
    throw std::invalid_argument(
        "In Weed::div(a, b, out), out size does not match input size!");
  }
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
  if (isAComplex && isBComplex) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_complex, gpu_complex);
#else
    cpu_complex(a, b, out);
#endif
  } else if (isAComplex) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_mixed_c_left, gpu_mixed_c_left);
#else
    cpu_mixed_c_left(a, b, out);
#endif
  } else if (isBComplex) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_mixed_c_right, gpu_mixed_c_right);
#else
    cpu_mixed_c_right(a, b, out);
#endif
  } else {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_real, gpu_real);
#else
    cpu_real(a, b, out);
#endif
  }
}

DivKernel div_kernel;

void div(const Tensor &a, const Tensor &b, Tensor &out) {
  div_kernel.div(a, b, out);
}
} // namespace Weed
