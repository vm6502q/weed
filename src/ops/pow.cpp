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
#include "ops/util.hpp"
#include "tensors/flat_tensors.hpp"

#define CPU_INIT(type, strg)                                                   \
  const tcapint I_a = a.stride[0U];                                            \
  const tcapint I_o = out.stride[0U];                                          \
  CAST_STORAGE(pa, a, type, strg);                                             \
  CAST_STORAGE(po, out, type, strg);                                           \
  size_t n = out.storage->size

#define GPU(type, p, api_call)                                                 \
  GPU_ARGS();                                                                  \
  std::shared_ptr<type> a_storage =                                            \
      std::dynamic_pointer_cast<type>(a.storage);                              \
  std::shared_ptr<type> o_storage =                                            \
      std::dynamic_pointer_cast<type>(out.storage);                            \
  const complex v = complex(p);                                                \
  a_storage->dev->RequestKernel(OCLAPI::api_call, args, a.get_size(),          \
                                {a_storage->buffer, o_storage->buffer}, 0U,    \
                                &v)

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
  const tcapint args[10U] {                                                    \
    a.offset, a.stride[0U], out.offset, out.stride[0U], 0U, 0U, 0U, 0U, 0U, 0U \
  }

namespace Weed {
static void cpu_real_pow(const Tensor &a, const real1 &p, Tensor &out) {
  CPU_INIT_2(RealTensor, RealTensor);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    po->write(i, (real1)std::pow((real1_s)(*pa)[i], (real1_s)p));
  };
  SPARSE_CPU_2_RUN(SparseCpuRealStorage);
}
static void cpu_real_exp(const Tensor &a, const real1 &b, Tensor &out) {
  CPU_INIT_2(RealTensor, RealTensor);
  const real1 log_b = (real1)std::log((real1_s)b);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    po->write(i, (real1)std::exp((real1_s)((*pa)[i] * log_b)));
  };
  SPARSE_CPU_2_RUN(SparseCpuRealStorage);
}
static void cpu_real_log(const Tensor &a, const real1 &b, Tensor &out) {
  CPU_INIT_2(RealTensor, RealTensor);
  const real1 inv_log_b = (real1)(ONE_R1 / std::log((real1_s)b));
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    po->write(i, (real1)(std::log((real1_s)(*pa)[i])) * inv_log_b);
  };
  SPARSE_CPU_2_RUN(SparseCpuRealStorage);
}
static void cpu_complex_pow(const Tensor &a, const real1 &p, Tensor &out) {
  CPU_INIT_2(ComplexTensor, ComplexTensor);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    po->write(i, std::pow((*pa)[i], p));
  };
  SPARSE_CPU_2_RUN(SparseCpuComplexStorage);
}
static void cpu_complex_exp(const Tensor &a, const real1 &b, Tensor &out) {
  CPU_INIT_2(ComplexTensor, ComplexTensor);
  const real1 log_b = (real1)std::log((real1_s)b);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    po->write(i, std::exp((*pa)[i] * log_b));
  };
  SPARSE_CPU_2_RUN(SparseCpuComplexStorage);
}
static void cpu_complex_log(const Tensor &a, const real1 &b, Tensor &out) {
  CPU_INIT_2(ComplexTensor, ComplexTensor);
  const real1 inv_log_b = (real1)(ONE_R1 / std::log((real1_s)b));
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    po->write(i, std::log((*pa)[i]) * inv_log_b);
  };
  SPARSE_CPU_2_RUN(SparseCpuComplexStorage);
}
#if ENABLE_GPU
static void gpu_real_pow(const Tensor &a, const real1 &p, Tensor &out) {
  GPU(GpuRealStorage, p, OCL_API_POW_REAL);
}
static void gpu_real_exp(const Tensor &a, const real1 &b, Tensor &out) {
  GPU(GpuRealStorage, (real1)std::log((real1_s)b), OCL_API_EXP_REAL);
}
static void gpu_real_log(const Tensor &a, const real1 &b, Tensor &out) {
  GPU(GpuRealStorage, (real1)(ONE_R1 / std::log((real1_s)b)), OCL_API_LOG_REAL);
}
static void gpu_complex_pow(const Tensor &a, const real1 &p, Tensor &out) {
  GPU(GpuComplexStorage, p, OCL_API_POW_COMPLEX);
}
static void gpu_complex_exp(const Tensor &a, const real1 &b, Tensor &out) {
  GPU(GpuComplexStorage, (real1)std::log((real1_s)b), OCL_API_EXP_COMPLEX);
}
static void gpu_complex_log(const Tensor &a, const real1 &b, Tensor &out) {
  GPU(GpuComplexStorage, (real1)(ONE_R1 / std::log((real1_s)b)),
      OCL_API_LOG_COMPLEX);
}
#endif
void PowKernel::pow(const Tensor &a, const real1 &p, Tensor &out) {
  validate_all_same_device({&a, &out}, "PowKernel::pow");
  const tcapint aSize = a.get_broadcast_size();
  const tcapint outSize = out.get_broadcast_size();
  if (aSize != outSize) {
    throw std::invalid_argument(
        "In Weed::pow(a, b, out), out size does not match input size!");
  }
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

PowKernel pow_kernel = {cpu_real_pow, cpu_complex_pow,
#if ENABLE_GPU
                        gpu_real_pow, gpu_complex_pow
#endif
};

PowKernel exp_kernel = {cpu_real_exp, cpu_complex_exp,
#if ENABLE_GPU
                        gpu_real_exp, gpu_complex_exp
#endif
};

PowKernel log_kernel = {cpu_real_log, cpu_complex_log,
#if ENABLE_GPU
                        gpu_real_log, gpu_complex_log
#endif
};

void pow(const Tensor &a, const real1 &p, Tensor &out) {
  pow_kernel.pow(a, p, out);
}
void exp(const Tensor &a, const real1 &b, Tensor &out) {
  exp_kernel.pow(a, b, out);
}
void log(const Tensor &a, const real1 &b, Tensor &out) {
  log_kernel.pow(a, b, out);
}
} // namespace Weed
