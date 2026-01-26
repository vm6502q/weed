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

#include "ops/abs.hpp"
#include "common/parallel_for.hpp"
#include "storage/all_storage.hpp"

#define DEVICE_SWITCH(cpu, gpu, a, out)                                        \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, out);                                                               \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, out);                                                               \
  }

#define DEVICE_SWITCH_GRAD(cpu, gpu, a, b, out)                                \
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

#define GPU_GRAD_ARGS()                                                        \
  const vecCapIntGpu args[10U] {                                               \
    (vecCapIntGpu)(din.offset), (vecCapIntGpu)(din.stride[0U]),                \
        (vecCapIntGpu)(in.offset), (vecCapIntGpu)(in.stride[0U]),              \
        (vecCapIntGpu)(dout.offset), (vecCapIntGpu)(dout.stride[0U]), 0U, 0U,  \
        0U, 0U                                                                 \
  }
namespace Weed {
void AbsKernel::cpu_real(const Tensor &a, Tensor &out) {
  const vecCapIntGpu I_a = (vecCapIntGpu)a.stride[0U];
  const vecCapIntGpu I_o = (vecCapIntGpu)out.stride[0U];
  CAST_STORAGE(pa, a, real1, CpuRealStorage);
  CAST_STORAGE(po, out, real1, CpuRealStorage);
  size_t n = out.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    po[i * I_o] = std::abs(pa[i * I_a]);
  });
}
void AbsKernel::cpu_complex(const Tensor &a, Tensor &out) {
  const vecCapIntGpu I_a = (vecCapIntGpu)(a.stride[0U]);
  const vecCapIntGpu I_o = (vecCapIntGpu)(out.stride[0U]);
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(po, out, real1, CpuRealStorage);
  size_t n = out.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    po[i * I_o] = std::abs(pa[i * I_a]);
  });
}
#if ENABLE_GPU
void AbsKernel::gpu_real(const Tensor &a, Tensor &out) {
  GPU_ARGS();
  GpuRealStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(a.storage);
  GpuRealStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(out.storage);
  a_storage->gpu->RequestKernel(OCLAPI::OCL_API_ABS_REAL, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer});
}
void AbsKernel::gpu_complex(const Tensor &a, Tensor &out) {
  GPU_ARGS();
  GpuComplexStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(a.storage);
  GpuRealStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(out.storage);
  a_storage->gpu->RequestKernel(OCLAPI::OCL_API_ABS_COMPLEX, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer});
}
#endif
void AbsKernel::abs(const Tensor &a, Tensor &out) {
  if (a.get_size() != out.get_size()) {
    throw std::invalid_argument(
        "In Weed::abs(a, out), out size does not match input size!");
  }
  switch (a.storage->dtype) {
  case DType::COMPLEX:
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_complex, gpu_complex, a, out)
#else
    cpu_complex(a, out);
#endif
    break;
  case DType::REAL:
  default:
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_real, gpu_real, a, out)
#else
    cpu_real(a, out);
#endif
  }
}

void AbsKernel::cpu_real_grad_real(Tensor &din, const Tensor &in,
                                   const Tensor &dout) {
  const vecCapIntGpu I_d = (vecCapIntGpu)(din.stride[0U]);
  const vecCapIntGpu I_i = (vecCapIntGpu)(in.stride[0U]);
  const vecCapIntGpu I_o = (vecCapIntGpu)(dout.stride[0U]);
  CAST_STORAGE(pdi, din, real1, CpuRealStorage);
  CAST_STORAGE(pi, in, real1, CpuRealStorage);
  CAST_STORAGE(po, dout, real1, CpuRealStorage);
  size_t n = dout.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    const real1 tmp = pi[i * I_i];
    const real1 tmp_o = po[i * I_o];
    pdi[i * I_d] +=
        (tmp == ZERO_R1) ? ZERO_R1 : ((tmp > ZERO_R1) ? tmp_o : -tmp_o);
  });
}
void AbsKernel::cpu_real_grad_complex(Tensor &din, const Tensor &in,
                                      const Tensor &dout) {
  const vecCapIntGpu I_d = (vecCapIntGpu)(din.stride[0U]);
  const vecCapIntGpu I_i = (vecCapIntGpu)(in.stride[0U]);
  const vecCapIntGpu I_o = (vecCapIntGpu)(dout.stride[0U]);
  CAST_STORAGE(pdi, din, complex, CpuComplexStorage);
  CAST_STORAGE(pi, in, real1, CpuRealStorage);
  CAST_STORAGE(po, dout, complex, CpuComplexStorage);
  size_t n = dout.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    const real1 tmp = pi[i * I_i];
    const complex tmp_o = po[i * I_o];
    pdi[i * I_d] +=
        (tmp == ZERO_R1) ? ZERO_CMPLX : ((tmp > ZERO_R1) ? tmp_o : -tmp_o);
  });
}
void AbsKernel::cpu_complex_grad_real(Tensor &din, const Tensor &in,
                                      const Tensor &dout) {
  const vecCapIntGpu I_d = (vecCapIntGpu)(din.stride[0U]);
  const vecCapIntGpu I_i = (vecCapIntGpu)(in.stride[0U]);
  const vecCapIntGpu I_o = (vecCapIntGpu)(dout.stride[0U]);
  CAST_STORAGE(pdi, din, complex, CpuComplexStorage);
  CAST_STORAGE(pi, in, complex, CpuComplexStorage);
  CAST_STORAGE(po, dout, real1, CpuRealStorage);
  size_t n = dout.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    const complex tmp = pi[i * I_i];
    const real1 tmp_o = po[i * I_o];
    const real1 out = std::abs(tmp);
    pdi[i * I_d] += (tmp == ZERO_CMPLX) ? ZERO_CMPLX : ((tmp_o / out) * tmp);
  });
}
void AbsKernel::cpu_complex_grad_complex(Tensor &din, const Tensor &in,
                                         const Tensor &dout) {
  const vecCapIntGpu I_d = (vecCapIntGpu)(din.stride[0U]);
  const vecCapIntGpu I_i = (vecCapIntGpu)(in.stride[0U]);
  const vecCapIntGpu I_o = (vecCapIntGpu)(dout.stride[0U]);
  CAST_STORAGE(pdi, din, complex, CpuComplexStorage);
  CAST_STORAGE(pi, in, complex, CpuComplexStorage);
  CAST_STORAGE(po, dout, complex, CpuComplexStorage);
  size_t n = dout.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    const complex tmp = pi[i * I_i];
    const complex tmp_o = po[i * I_o];
    const real1 out = std::abs(tmp);
    pdi[i * I_d] += (tmp == ZERO_CMPLX) ? ZERO_CMPLX : (tmp_o * tmp / out);
  });
}
#if ENABLE_GPU
void AbsKernel::gpu_real_grad_real(Tensor &din, const Tensor &in,
                                   const Tensor &dout) {
  GPU_GRAD_ARGS();
  GpuRealStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(din.storage);
  GpuRealStoragePtr b_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(in.storage);
  GpuRealStoragePtr c_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(dout.storage);
  a_storage->gpu->RequestKernel(
      OCLAPI::OCL_API_ABS_REAL_GRAD_REAL, args, din.get_size(),
      {a_storage->buffer, b_storage->buffer, c_storage->buffer});
}
void AbsKernel::gpu_real_grad_complex(Tensor &din, const Tensor &in,
                                      const Tensor &dout) {
  GPU_GRAD_ARGS();
  GpuComplexStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(din.storage);
  GpuRealStoragePtr b_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(in.storage);
  GpuComplexStoragePtr c_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(dout.storage);
  a_storage->gpu->RequestKernel(
      OCLAPI::OCL_API_ABS_REAL_GRAD_COMPLEX, args, din.get_size(),
      {a_storage->buffer, b_storage->buffer, c_storage->buffer});
}
void AbsKernel::gpu_complex_grad_real(Tensor &din, const Tensor &in,
                                      const Tensor &dout) {
  GPU_GRAD_ARGS();
  GpuComplexStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(din.storage);
  GpuComplexStoragePtr b_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(in.storage);
  GpuRealStoragePtr c_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(dout.storage);
  a_storage->gpu->RequestKernel(
      OCLAPI::OCL_API_ABS_COMPLEX_GRAD_REAL, args, din.get_size(),
      {a_storage->buffer, b_storage->buffer, c_storage->buffer});
}
void AbsKernel::gpu_complex_grad_complex(Tensor &din, const Tensor &in,
                                         const Tensor &dout) {
  GPU_GRAD_ARGS();
  GpuComplexStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(din.storage);
  GpuComplexStoragePtr b_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(in.storage);
  GpuComplexStoragePtr c_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(dout.storage);
  a_storage->gpu->RequestKernel(
      OCLAPI::OCL_API_ABS_COMPLEX_GRAD_COMPLEX, args, din.get_size(),
      {a_storage->buffer, b_storage->buffer, c_storage->buffer});
}
#endif
void AbsKernel::abs_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
  if (din.storage->dtype != dout.storage->dtype) {
    throw std::invalid_argument("In Weed::abs_grad(din, in, dout), din dtype "
                                "and dout dtype must match!");
  }
  const size_t dinSize = din.get_size();
  const size_t inSize = in.get_size();
  const size_t doutSize = dout.get_size();
  if ((dinSize != inSize) || (dinSize != doutSize)) {
    throw std::invalid_argument(
        "In Weed::abs_grad(din, in, dout), sizes do not match!");
  }
  switch (in.storage->dtype) {
  case DType::COMPLEX:
    switch (dout.storage->dtype) {
    case DType::COMPLEX:
#if ENABLE_GPU
      DEVICE_SWITCH_GRAD(cpu_complex_grad_complex, gpu_complex_grad_complex,
                         din, in, dout)
#else
      cpu_complex_grad_complex(din, in, dout);
#endif
      break;
    case DType::REAL:
    default:
#if ENABLE_GPU
      DEVICE_SWITCH_GRAD(cpu_complex_grad_real, gpu_complex_grad_real, din, in,
                         dout);
#else
      cpu_complex_grad_real(din, in, dout);
#endif
    }
    break;
  case DType::REAL:
  default:
    switch (dout.storage->dtype) {
    case DType::COMPLEX:
#if ENABLE_GPU
      DEVICE_SWITCH_GRAD(cpu_real_grad_complex, gpu_real_grad_complex, din, in,
                         dout);
#else
      cpu_real_grad_complex(din, in, dout);
#endif
      break;
    case DType::REAL:
    default:
#if ENABLE_GPU
      DEVICE_SWITCH_GRAD(cpu_real_grad_real, gpu_real_grad_real, din, in, dout);
#else
      cpu_real_grad_real(din, in, dout);
#endif
    }
  }
}

AbsKernel abs_kernel;

void abs(const Tensor &a, Tensor &out) { abs_kernel.abs(a, out); }
void abs_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
  abs_kernel.abs_grad(din, in, dout);
}
} // namespace Weed
