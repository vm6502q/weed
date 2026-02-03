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
#include "ops/util.hpp"
#include "tensors/flat_tensors.hpp"

#define DEVICE_SWITCH(cpu, gpu, a, out)                                        \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, out);                                                               \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, out);                                                               \
  }

#define CPU_KERNEL(type, storage)                                              \
  const unsigned cpuCount =                                                    \
      (unsigned)std::min(n, (size_t)pfControl.GetNumCores());                  \
  std::vector<type> total(cpuCount, ZERO_R1);                                  \
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    total[cpu] += (*pa)[i];                                                    \
  };                                                                           \
  SPARSE_CPU_2_RUN(storage);                                                   \
  type &t = total[0U];                                                         \
  for (size_t i = 1U; i < cpuCount; ++i) {                                     \
    t += total[i];                                                             \
  }

#define CPU_SUM(type)                                                          \
  const unsigned cpuCount =                                                    \
      (unsigned)std::min(n, (size_t)pfControl.GetNumCores());                  \
  std::vector<type> total(cpuCount, ZERO_R1);                                  \
  pfControl.par_for(0, n, [&](const tcapint &i, const unsigned &cpu) {         \
    total[cpu] += (*pa)[i];                                                    \
  });                                                                          \
  type &t = total[0U];                                                         \
  for (size_t i = 1U; i < cpuCount; ++i) {                                     \
    t += total[i];                                                             \
  }

#define GPU_SUM(type)                                                          \
  if (!(a_storage->data)) {                                                    \
    a_storage->data = a_storage->Alloc(n);                                     \
  }                                                                            \
  type *gpa = a_storage->data.get();                                           \
  const bool isMapped =                                                        \
      a_storage->dev->LockSync(a_storage->buffer, sizeof(type) * n, gpa);      \
  CPU_SUM(type)

#define GPU_WRITE(SetType)                                                     \
  o_storage->dev->SetType(t, o_storage->buffer, 0U);                           \
  if (isMapped) {                                                              \
    a_storage->dev->UnlockSync(a_storage->buffer, a_storage->data.get());      \
  } else {                                                                     \
    a_storage->data = nullptr;                                                 \
  }

#define GPU_CAST(storage1, storage2)                                           \
  storage1 *a_storage = static_cast<storage1 *>(a.storage.get());              \
  storage2 *o_storage = static_cast<storage2 *>(out.storage.get())

namespace Weed {
static void cpu_sum_real(const Tensor &a, Tensor &out) {
  CPU_INIT_2_SCALAR(RealTensor, RealStorage);
  CPU_KERNEL(real1, SparseCpuRealStorage);
  po->write(0U, t);
}
static void cpu_mean_real(const Tensor &a, Tensor &out) {
  cpu_sum_real(a, out);
  GET_STORAGE(RealStorage, out, po);
  po->write(0U, (*po)[0U] / (real1)a.get_broadcast_size());
}
static void cpu_sum_complex(const Tensor &a, Tensor &out) {
  CPU_INIT_2_SCALAR(ComplexTensor, ComplexStorage);
  CPU_KERNEL(complex, SparseCpuComplexStorage);
  po->write(0U, t);
}
static void cpu_mean_complex(const Tensor &a, Tensor &out) {
  cpu_sum_complex(a, out);
  GET_STORAGE(ComplexStorage, out, po);
  po->write(0U, (*po)[0U] / (real1)a.get_broadcast_size());
}
#if ENABLE_GPU
static void gpu_sum_real(const Tensor &a, Tensor &out) {
  GPU_INIT_2_SCALAR(RealStorage, RealStorage);
  GPU_CAST(GpuRealStorage, GpuRealStorage);
  GPU_SUM(real1);
  GPU_WRITE(SetReal);
}
static void gpu_mean_real(const Tensor &a, Tensor &out) {
  GPU_INIT_2_SCALAR(RealStorage, RealStorage);
  GPU_CAST(GpuRealStorage, GpuRealStorage);
  GPU_SUM(real1);
  t /= n;
  GPU_WRITE(SetReal);
}
static void gpu_sum_complex(const Tensor &a, Tensor &out) {
  GPU_INIT_2_SCALAR(ComplexStorage, ComplexStorage);
  GPU_CAST(GpuComplexStorage, GpuComplexStorage);
  GPU_SUM(complex);
  GPU_WRITE(SetComplex);
}
static void gpu_mean_complex(const Tensor &a, Tensor &out) {
  GPU_INIT_2_SCALAR(ComplexStorage, ComplexStorage);
  GPU_CAST(GpuComplexStorage, GpuComplexStorage);
  GPU_SUM(complex);
  t /= n;
  GPU_WRITE(SetComplex);
}
#endif
void SumKernel::sum(const Tensor &a, Tensor &out) {
  validate_all_same_device({&a, &out}, "ClampKernel::clamp");
  if (out.get_broadcast_size() != 1U) {
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
