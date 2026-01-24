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

#include "relu.hpp"
#include "common/parallel_for.hpp"
#include "cpu_complex_storage.hpp"
#include "cpu_real_storage.hpp"
#if ENABLE_GPU
#include "gpu_complex_storage.hpp"
#include "gpu_real_storage.hpp"
#endif

#define CAST_STORAGE(out, in, type, ptr)                                       \
  type *out = static_cast<ptr *>(in.storage.get())->data.get() + in.offset

#define DEVICE_SWITCH(cpu, gpu, a, b, out)                                     \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, b, out);                                                            \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, b, out);                                                            \
  }

#define GPU_GRAD_ARGS()                                                        \
  const vecCapIntGpu args[10U] {                                               \
    (vecCapIntGpu)(din.offset), (vecCapIntGpu)(din.stride[0U]),                \
        (vecCapIntGpu)(in.offset), (vecCapIntGpu)(in.stride[0U]),              \
        (vecCapIntGpu)(dout.offset), (vecCapIntGpu)(dout.stride[0U]), 0U, 0U,  \
        0U, 0U                                                                 \
  }

namespace Weed {
void ReluKernel::cpu(const Tensor &a, Tensor &out) {
  const vecCapIntGpu I_a = (vecCapIntGpu)(a.stride[0U]);
  const vecCapIntGpu I_o = (vecCapIntGpu)(out.stride[0U]);
  CAST_STORAGE(pa, a, real1, CpuRealStorage);
  CAST_STORAGE(po, out, real1, CpuRealStorage);
  size_t n = out.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    po[i * I_o] = std::max(pa[i * I_a], ZERO_R1);
  });
}
#if ENABLE_GPU
void ReluKernel::gpu(const Tensor &a, Tensor &out) {
  const vecCapIntGpu args[10U]{(vecCapIntGpu)(a.offset),
                               (vecCapIntGpu)(a.stride[0U]),
                               (vecCapIntGpu)(out.offset),
                               (vecCapIntGpu)(out.stride[0U]),
                               0U,
                               0U,
                               0U,
                               0U,
                               0U,
                               0U};
  GpuRealStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(a.storage);
  GpuRealStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(out.storage);
  a_storage->gpu->RequestKernel(OCLAPI::OCL_API_RELU, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer});
}
#endif
void ReluKernel::relu(const Tensor &a, Tensor &out) {
  if ((a.storage->dtype == DType::COMPLEX) or
      (out.storage->dtype == DType::COMPLEX)) {
    throw std::invalid_argument("Cannot apply ReLU on complex tensors!");
  }
#if ENABLE_GPU
  switch (out.storage->device) {
  case DeviceTag::GPU:
    gpu(a, out);
    break;
  case DeviceTag::CPU:
  default:
    cpu(a, out);
  }
#else
  cpu(a, out);
#endif
}

void ReluKernel::cpu_grad_real(Tensor &din, const Tensor &in,
                               const Tensor &dout) {
  const vecCapIntGpu I_d = (vecCapIntGpu)(din.stride[0U]);
  const vecCapIntGpu I_i = (vecCapIntGpu)(in.stride[0U]);
  const vecCapIntGpu I_o = (vecCapIntGpu)(dout.stride[0U]);
  CAST_STORAGE(pdi, din, real1, CpuRealStorage);
  CAST_STORAGE(pi, in, real1, CpuRealStorage);
  CAST_STORAGE(po, dout, real1, CpuRealStorage);
  size_t n = dout.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    pdi[i * I_d] = (pi[i * I_i] > 0) ? po[i * I_o] : ZERO_R1;
  });
}
void ReluKernel::cpu_grad_complex(Tensor &din, const Tensor &in,
                                  const Tensor &dout) {
  const vecCapIntGpu I_d = (vecCapIntGpu)(din.stride[0U]);
  const vecCapIntGpu I_i = (vecCapIntGpu)(in.stride[0U]);
  const vecCapIntGpu I_o = (vecCapIntGpu)(dout.stride[0U]);
  CAST_STORAGE(pdi, din, complex, CpuComplexStorage);
  CAST_STORAGE(pi, in, real1, CpuRealStorage);
  CAST_STORAGE(po, dout, complex, CpuComplexStorage);
  size_t n = dout.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    pdi[i * I_d] = (pi[i * I_i] > 0) ? po[i * I_o] : ZERO_CMPLX;
  });
}
#if ENABLE_GPU
void ReluKernel::gpu_grad_real(Tensor &din, const Tensor &in,
                               const Tensor &dout) {
  GPU_GRAD_ARGS();
  GpuRealStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(din.storage);
  GpuRealStoragePtr b_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(in.storage);
  GpuRealStoragePtr c_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(dout.storage);
  a_storage->gpu->RequestKernel(
      OCLAPI::OCL_API_RELU_GRAD_REAL, args, din.get_size(),
      {a_storage->buffer, b_storage->buffer, c_storage->buffer});
}
void ReluKernel::gpu_grad_complex(Tensor &din, const Tensor &in,
                                  const Tensor &dout) {
  GPU_GRAD_ARGS();
  GpuComplexStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(din.storage);
  GpuRealStoragePtr b_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(in.storage);
  GpuComplexStoragePtr c_storage =
      std::dynamic_pointer_cast<GpuComplexStorage>(dout.storage);
  a_storage->gpu->RequestKernel(
      OCLAPI::OCL_API_RELU_GRAD_COMPLEX, args, din.get_size(),
      {a_storage->buffer, b_storage->buffer, c_storage->buffer});
}
#endif
void ReluKernel::relu_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
  switch (din.storage->dtype) {
  case DType::COMPLEX:
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_grad_complex, gpu_grad_complex, din, in, dout);
#else
    cpu_grad_complex(din, in, dout);
#endif
    break;
  case DType::REAL:
  default:
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_grad_real, gpu_grad_real, din, in, dout);
#else
    cpu_grad_real(din, in, dout);
#endif
  }
}

ReluKernel relu_kernel;

void relu(const Tensor &a, Tensor &out) { relu_kernel.relu(a, out); }
void relu_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
  relu_kernel.relu_grad(din, in, dout);
}
} // namespace Weed
