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

#include "ops/pow.hpp"
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

#define GPU_ARGS()                                                             \
  const vecCapIntGpu args[10U] {                                               \
    (vecCapIntGpu)(a.offset), (vecCapIntGpu)(a.stride[0U]),                    \
        (vecCapIntGpu)(out.offset), (vecCapIntGpu)(out.stride[0U]), 0U, 0U,    \
        0U, 0U, 0U, 0U                                                         \
  }

namespace Weed {
void PowKernel::cpu_real(const Tensor &a, const RealScalar &p, Tensor &out) {
  const vecCapIntGpu I_a = (vecCapIntGpu)a.stride[0U];
  const vecCapIntGpu I_o = (vecCapIntGpu)out.stride[0U];
  CAST_STORAGE(pa, a, real1, CpuRealStorage);
  CAST_STORAGE(po, out, real1, CpuRealStorage);
  const real1 v = p.get_item();
  size_t n = out.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    po[i * I_o] = std::pow(pa[i * I_a], v);
  });
}
void PowKernel::cpu_complex(const Tensor &a, const RealScalar &p, Tensor &out) {
  const vecCapIntGpu I_a = (vecCapIntGpu)(a.stride[0U]);
  const vecCapIntGpu I_o = (vecCapIntGpu)(out.stride[0U]);
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(po, out, complex, CpuComplexStorage);
  const real1 v = p.get_item();
  size_t n = out.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    po[i * I_o] = std::pow(pa[i * I_a], v);
  });
}
#if ENABLE_GPU
void PowKernel::gpu_real(const Tensor &a, const RealScalar &p, Tensor &out) {
  GPU_ARGS();
  GpuRealStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(a.storage);
  GpuRealStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(out.storage);
  const complex v = complex(p.get_item());
  a_storage->gpu->RequestKernel(OCLAPI::OCL_API_POW_REAL, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer}, 0U, &v);
}
void PowKernel::gpu_complex(const Tensor &a, const RealScalar &p, Tensor &out) {
  GPU_ARGS();
  GpuComplexStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(a.storage);
  GpuComplexStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(out.storage);
  const complex v = complex(p.get_item());
  a_storage->gpu->RequestKernel(OCLAPI::OCL_API_POW_COMPLEX, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer}, 0U, &v);
}
#endif
void PowKernel::pow(const Tensor &a, const RealScalar &p, Tensor &out) {
  switch (a.storage->dtype) {
  case DType::COMPLEX:
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_complex, gpu_complex, a, p, out)
#else
    cpu_complex(a, p, out);
#endif
    break;
  case DType::REAL:
  default:
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_real, gpu_real, a, p, out)
#else
    cpu_real(a, p, out);
#endif
  }
}

PowKernel pow_kernel;

void pow(const Tensor &a, const RealScalar &p, Tensor &out) {
  pow_kernel.pow(a, p, out);
}
} // namespace Weed
