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

#include "mean.hpp"
#include "common/parallel_for.hpp"
#include "cpu_complex_storage.hpp"
#include "cpu_real_storage.hpp"
#if ENABLE_GPU
#include "gpu_complex_storage.hpp"
#include "gpu_real_storage.hpp"
#endif

#define CAST_STORAGE(out, in, type, ptr)                                       \
  type *out = static_cast<ptr *>(in.storage.get())->data.get() + in.offset

#define DEVICE_SWITCH(cpu, gpu, a, out)                                        \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, out);                                                               \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, out);                                                               \
  }

#define CPU_SUM(type)                                                          \
  unsigned cpuCount = pfControl.GetNumCores();                                 \
  std::vector<type> total(cpuCount, ZERO_R1);                                  \
  pfControl.par_for(0, sz, [&](const vecCapIntGpu &i, const unsigned &cpu) {   \
    total[cpu] += pa[i * I_a];                                                 \
  });                                                                          \
  type t = ZERO_R1;                                                            \
  for (size_t i = 0U; i < cpuCount; ++i) {                                     \
    t += total[i];                                                             \
  }                                                                            \
  t /= sz

#define GPU_SUM(type, SetType)                                                 \
  const vecCapIntGpu I_a = (vecCapIntGpu)a.stride[0U];                         \
  size_t sz = a_storage->size;                                                 \
  if (!(a_storage->array)) {                                                   \
    a_storage->array = a_storage->Alloc(sz);                                   \
  }                                                                            \
  type *pa = a_storage->array.get();                                           \
  const bool isMapped =                                                        \
      a_storage->gpu->LockSync(a_storage->buffer, sizeof(type) * sz, pa);      \
  CPU_SUM(type);                                                               \
  o_storage->gpu->SetType(t, o_storage->buffer, 0U);                           \
  if (isMapped) {                                                              \
    a_storage->gpu->UnlockSync(a_storage->buffer, a_storage->array.get());     \
  } else {                                                                     \
    a_storage->array = nullptr;                                                \
  }

namespace Weed {
void MeanKernel::cpu_real(const Tensor &a, Tensor &out) {
  const vecCapIntGpu I_a = (vecCapIntGpu)a.stride[0U];
  CAST_STORAGE(pa, a, real1, CpuRealStorage);
  CAST_STORAGE(po, out, real1, CpuRealStorage);
  size_t sz = a.storage->size;
  CPU_SUM(real1);
  po[0U] = t;
}
void MeanKernel::cpu_complex(const Tensor &a, Tensor &out) {
  const vecCapIntGpu I_a = (vecCapIntGpu)(a.stride[0U]);
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(po, out, complex, CpuComplexStorage);
  size_t sz = a.storage->size;
  CPU_SUM(complex);
  po[0U] = t;
}
#if ENABLE_GPU
void MeanKernel::gpu_real(const Tensor &a, Tensor &out) {
  GpuRealStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(a.storage);
  GpuRealStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(out.storage);
  GPU_SUM(real1, SetReal);
}
void MeanKernel::gpu_complex(const Tensor &a, Tensor &out) {
  GpuComplexStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(a.storage);
  GpuComplexStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(out.storage);
  GPU_SUM(complex, SetComplex);
}
#endif
void MeanKernel::mean(const Tensor &a, Tensor &out) {
  switch (a.storage->dtype) {
  case DType::COMPLEX:
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_complex, gpu_complex, a, out);
#else
    cpu_complex(a, out);
#endif
    break;
  case DType::REAL:
  default:
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_real, gpu_real, a, out);
#else
    cpu_real(a, out);
#endif
  }
}

MeanKernel mean_kernel;

void mean(const Tensor &a, Tensor &out) { mean_kernel.mean(a, out); }
} // namespace Weed
