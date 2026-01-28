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

#include "ops/sum.hpp"
#include "common/parallel_for.hpp"
#include "storage/all_storage.hpp"

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
  pfControl.par_for(0, n, [&](const tcapint &i, const unsigned &cpu) {         \
    total[cpu] += pa[O_a + i * I_a];                                           \
  });                                                                          \
  type t = ZERO_R1;                                                            \
  for (size_t i = 0U; i < cpuCount; ++i) {                                     \
    t += total[i];                                                             \
  }

#define GPU_SUM(type)                                                          \
  if (!(a_storage.array)) {                                                    \
    a_storage.array = a_storage.Alloc(n);                                      \
  }                                                                            \
  type *gpa = a_storage.array.get();                                           \
  const bool isMapped =                                                        \
      a_storage.dev->LockSync(a_storage.buffer, sizeof(type) * n, gpa);        \
  CPU_SUM(type);

#define GPU_WRITE(SetType)                                                     \
  o_storage.dev->SetType(t, o_storage.buffer, 0U);                             \
  if (isMapped) {                                                              \
    a_storage.dev->UnlockSync(a_storage.buffer, a_storage.array.get());        \
  } else {                                                                     \
    a_storage.array = nullptr;                                                 \
  }

namespace Weed {
static void cpu_sum_real(const Tensor &a, Tensor &out) {
  CPU_INIT_2_SCALAR(CpuRealStorage, CpuRealStorage);
  CPU_SUM(real1);
  po.write(0U, t);
}
static void cpu_mean_real(const Tensor &a, Tensor &out) {
  cpu_sum_real(a, out);
  CAST_STORAGE(po, out, real1, CpuRealStorage);
  po[0U] /= a.get_size();
}
static void cpu_sum_complex(const Tensor &a, Tensor &out) {
  CPU_INIT_2_SCALAR(CpuComplexStorage, CpuComplexStorage);
  CPU_SUM(complex);
  po.write(0U, t);
}
static void cpu_mean_complex(const Tensor &a, Tensor &out) {
  cpu_sum_complex(a, out);
  CAST_STORAGE(po, out, complex, CpuComplexStorage);
  po[0U] /= (real1)a.get_size();
}
#if ENABLE_GPU
static void gpu_sum_real(const Tensor &a, Tensor &out) {
  CPU_INIT_2_SCALAR(GpuRealStorage, GpuRealStorage);
  GpuRealStorage a_storage = *static_cast<GpuRealStorage *>(a.storage.get());
  GpuRealStorage o_storage = *static_cast<GpuRealStorage *>(out.storage.get());
  GPU_SUM(real1);
  GPU_WRITE(SetReal);
}
static void gpu_mean_real(const Tensor &a, Tensor &out) {
  CPU_INIT_2_SCALAR(GpuRealStorage, GpuRealStorage);
  GpuRealStorage a_storage = *static_cast<GpuRealStorage *>(a.storage.get());
  GpuRealStorage o_storage = *static_cast<GpuRealStorage *>(out.storage.get());
  GPU_SUM(real1);
  t /= n;
  GPU_WRITE(SetReal);
}
static void gpu_sum_complex(const Tensor &a, Tensor &out) {
  CPU_INIT_2_SCALAR(GpuComplexStorage, GpuComplexStorage);
  GpuComplexStorage a_storage =
      *static_cast<GpuComplexStorage *>(a.storage.get());
  GpuComplexStorage o_storage =
      *static_cast<GpuComplexStorage *>(out.storage.get());
  GPU_SUM(complex);
  GPU_WRITE(SetComplex);
}
static void gpu_mean_complex(const Tensor &a, Tensor &out) {
  CPU_INIT_2_SCALAR(GpuComplexStorage, GpuComplexStorage);
  GpuComplexStorage a_storage =
      *static_cast<GpuComplexStorage *>(a.storage.get());
  GpuComplexStorage o_storage =
      *static_cast<GpuComplexStorage *>(out.storage.get());
  GPU_SUM(complex);
  t /= n;
  GPU_WRITE(SetComplex);
}
#endif
void SumKernel::sum(const Tensor &a, Tensor &out) {
  if (out.get_size() != 1U) {
    throw std::invalid_argument("In Weed::sum(a, out) or Weed::mean(a, out), "
                                "out parameter is not a scalar!");
  }
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

SumKernel sum_kernel = {cpu_sum_real, cpu_sum_complex,
#if ENABLE_GPU
                        gpu_sum_real, gpu_sum_complex
#endif
};
SumKernel mean_kernel = {cpu_mean_real, cpu_mean_complex,
#if ENABLE_GPU
                         gpu_mean_real, gpu_mean_complex
#endif
};

void sum(const Tensor &a, Tensor &out) { sum_kernel.sum(a, out); }
void mean(const Tensor &a, Tensor &out) { mean_kernel.sum(a, out); }
} // namespace Weed
