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
static void cpu_real_pow(const Tensor &a, const real1 &p, Tensor &out) {
  const vecCapIntGpu I_a = (vecCapIntGpu)a.stride[0U];
  const vecCapIntGpu I_o = (vecCapIntGpu)out.stride[0U];
  CAST_STORAGE(pa, a, real1, CpuRealStorage);
  CAST_STORAGE(po, out, real1, CpuRealStorage);
  size_t n = out.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    po[i * I_o] = std::pow(pa[i * I_a], p);
  });
}
static void cpu_real_exp(const Tensor &a, const real1 &b, Tensor &out) {
  const vecCapIntGpu I_a = (vecCapIntGpu)a.stride[0U];
  const vecCapIntGpu I_o = (vecCapIntGpu)out.stride[0U];
  CAST_STORAGE(pa, a, real1, CpuRealStorage);
  CAST_STORAGE(po, out, real1, CpuRealStorage);
  const real1 log_b = (real1)std::log(b);
  size_t n = out.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    po[i * I_o] = std::exp(pa[i * I_a] * log_b);
  });
}
static void cpu_real_log(const Tensor &a, const real1 &b, Tensor &out) {
  const vecCapIntGpu I_a = (vecCapIntGpu)a.stride[0U];
  const vecCapIntGpu I_o = (vecCapIntGpu)out.stride[0U];
  CAST_STORAGE(pa, a, real1, CpuRealStorage);
  CAST_STORAGE(po, out, real1, CpuRealStorage);
  const real1 inv_log_b = (real1)(1.0 / std::log(b));
  size_t n = out.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    po[i * I_o] = std::log(pa[i * I_a]) * inv_log_b;
  });
}
static void cpu_complex_pow(const Tensor &a, const real1 &p, Tensor &out) {
  const vecCapIntGpu I_a = (vecCapIntGpu)(a.stride[0U]);
  const vecCapIntGpu I_o = (vecCapIntGpu)(out.stride[0U]);
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(po, out, complex, CpuComplexStorage);
  size_t n = out.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    po[i * I_o] = std::pow(pa[i * I_a], p);
  });
}
static void cpu_complex_exp(const Tensor &a, const real1 &b, Tensor &out) {
  const vecCapIntGpu I_a = (vecCapIntGpu)(a.stride[0U]);
  const vecCapIntGpu I_o = (vecCapIntGpu)(out.stride[0U]);
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(po, out, complex, CpuComplexStorage);
  const real1 log_b = (real1)std::log(b);
  size_t n = out.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    po[i * I_o] = std::exp(pa[i * I_a] * log_b);
  });
}
static void cpu_complex_log(const Tensor &a, const real1 &b, Tensor &out) {
  const vecCapIntGpu I_a = (vecCapIntGpu)(a.stride[0U]);
  const vecCapIntGpu I_o = (vecCapIntGpu)(out.stride[0U]);
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(po, out, complex, CpuComplexStorage);
  const real1 inv_log_b = (real1)(1.0 / std::log(b));
  size_t n = out.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    po[i * I_o] = std::log(pa[i * I_a]) * inv_log_b;
  });
}
#if ENABLE_GPU
static void gpu_real_pow(const Tensor &a, const real1 &p, Tensor &out) {
  GPU_ARGS();
  GpuRealStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(a.storage);
  GpuRealStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(out.storage);
  const complex v = complex(p);
  a_storage->gpu->RequestKernel(OCLAPI::OCL_API_POW_REAL, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer}, 0U, &v);
}
static void gpu_real_exp(const Tensor &a, const real1 &b, Tensor &out) {
  GPU_ARGS();
  GpuRealStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(a.storage);
  GpuRealStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(out.storage);
  const complex v = complex((real1)std::log(b));
  a_storage->gpu->RequestKernel(OCLAPI::OCL_API_EXP_REAL, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer}, 0U, &v);
}
static void gpu_real_log(const Tensor &a, const real1 &b, Tensor &out) {
  GPU_ARGS();
  GpuRealStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(a.storage);
  GpuRealStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(out.storage);
  const complex v = complex((real1)(1.0 / std::log(b)));
  a_storage->gpu->RequestKernel(OCLAPI::OCL_API_LOG_REAL, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer}, 0U, &v);
}
static void gpu_complex_pow(const Tensor &a, const real1 &p, Tensor &out) {
  GPU_ARGS();
  GpuComplexStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(a.storage);
  GpuComplexStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(out.storage);
  const complex v = complex(p);
  a_storage->gpu->RequestKernel(OCLAPI::OCL_API_POW_COMPLEX, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer}, 0U, &v);
}
static void gpu_complex_exp(const Tensor &a, const real1 &b, Tensor &out) {
  GPU_ARGS();
  GpuComplexStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(a.storage);
  GpuComplexStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(out.storage);
  const complex v = complex((real1)std::log(b));
  a_storage->gpu->RequestKernel(OCLAPI::OCL_API_EXP_COMPLEX, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer}, 0U, &v);
}
static void gpu_complex_log(const Tensor &a, const real1 &b, Tensor &out) {
  GPU_ARGS();
  GpuComplexStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(a.storage);
  GpuComplexStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(out.storage);
  const complex v = complex((real1)(1.0 / std::log(b)));
  a_storage->gpu->RequestKernel(OCLAPI::OCL_API_LOG_COMPLEX, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer}, 0U, &v);
}
#endif
void PowKernel::pow(const Tensor &a, const real1 &p, Tensor &out) {
  const size_t aSize = a.get_broadcast_size();
  const size_t outSize = out.get_broadcast_size();
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
