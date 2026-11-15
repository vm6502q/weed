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





namespace Weed {
template <typename T1, typename T2, typename T3>
static void cpu_sum(const Tensor &a, Tensor &out) {
  CPU_INIT_2_SCALAR(T1, T1);
  CPU_KERNEL(T3, T2);
  po->write(0U, t);
}
static inline void cpu_sum_real(const Tensor &a, Tensor &out) {
  cpu_sum<RealTensor, SparseCpuRealStorage, real1>(a, out);
}
static inline void cpu_sum_complex(const Tensor &a, Tensor &out) {
  cpu_sum<ComplexTensor, SparseCpuComplexStorage, complex>(a, out);
}

template <typename T1, typename T2, typename T3>
static void cpu_mean(const Tensor &a, Tensor &out) {
  cpu_sum<T1, T2, T3>(a, out);
  GET_STORAGE(T2, out, po);
  po->write(0U, (*po)[0U] / (real1)a.get_broadcast_size());
}
static inline void cpu_mean_real(const Tensor &a, Tensor &out) {
  cpu_mean<RealTensor, SparseCpuRealStorage, real1>(a, out);
}
static inline void cpu_mean_complex(const Tensor &a, Tensor &out) {
  cpu_mean<ComplexTensor, SparseCpuComplexStorage, complex>(a, out);
}
#if ENABLE_GPU
// Copy the tensor to host memory in one read, reduce it with the CPU kernel,
// and write the single result back to the device.
static void gpu_sum_real(const Tensor &a, Tensor &out) {
  const TensorPtr ca = a.cast(DeviceTag::CPU);
  Tensor co(std::vector<real1>{ZERO_R1}, std::vector<tcapint>{1U}, false,
            DeviceTag::CPU);
  cpu_sum_real(*ca, co);
  const real1 t = (*static_cast<RealStorage *>(co.storage.get()))[0U];
  GpuRealStorage *o_storage = static_cast<GpuRealStorage *>(out.storage.get());
  o_storage->dev->SetReal(t, o_storage->buffer, out.offset);
}
static void gpu_mean_real(const Tensor &a, Tensor &out) {
  const TensorPtr ca = a.cast(DeviceTag::CPU);
  Tensor co(std::vector<real1>{ZERO_R1}, std::vector<tcapint>{1U}, false,
            DeviceTag::CPU);
  cpu_mean_real(*ca, co);
  const real1 t = (*static_cast<RealStorage *>(co.storage.get()))[0U];
  GpuRealStorage *o_storage = static_cast<GpuRealStorage *>(out.storage.get());
  o_storage->dev->SetReal(t, o_storage->buffer, out.offset);
}
static void gpu_sum_complex(const Tensor &a, Tensor &out) {
  const TensorPtr ca = a.cast(DeviceTag::CPU);
  Tensor co(std::vector<complex>{ZERO_CMPLX}, std::vector<tcapint>{1U}, false,
            DeviceTag::CPU);
  cpu_sum_complex(*ca, co);
  const complex t = (*static_cast<ComplexStorage *>(co.storage.get()))[0U];
  GpuComplexStorage *o_storage =
      static_cast<GpuComplexStorage *>(out.storage.get());
  o_storage->dev->SetComplex(t, o_storage->buffer, out.offset);
}
static void gpu_mean_complex(const Tensor &a, Tensor &out) {
  const TensorPtr ca = a.cast(DeviceTag::CPU);
  Tensor co(std::vector<complex>{ZERO_CMPLX}, std::vector<tcapint>{1U}, false,
            DeviceTag::CPU);
  cpu_mean_complex(*ca, co);
  const complex t = (*static_cast<ComplexStorage *>(co.storage.get()))[0U];
  GpuComplexStorage *o_storage =
      static_cast<GpuComplexStorage *>(out.storage.get());
  o_storage->dev->SetComplex(t, o_storage->buffer, out.offset);
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
