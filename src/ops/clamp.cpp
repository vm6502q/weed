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

#include "ops/clamp.hpp"
#include "common/parallel_for.hpp"
#include "ops/util.hpp"
#include "tensors/flat_tensors.hpp"

#define GPU_GRAD(type1, type2, type3, api_call)                                \
  GPU_GRAD_ARGS();                                                             \
  std::shared_ptr<type1> a_storage =                                           \
      std::dynamic_pointer_cast<type1>(din.storage);                           \
  std::shared_ptr<type2> b_storage =                                           \
      std::dynamic_pointer_cast<type2>(in.storage);                            \
  std::shared_ptr<type3> c_storage =                                           \
      std::dynamic_pointer_cast<type3>(dout.storage);                          \
  const complex v = complex(l, h);                                             \
  a_storage->dev->RequestKernel(                                               \
      OCLAPI::api_call, args, din.get_size(),                                  \
      {a_storage->buffer, b_storage->buffer, c_storage->buffer}, 0U, &v)

#define DEVICE_SWITCH(cpu, gpu, a, l, h, out)                                  \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, l, h, out);                                                         \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, l, h, out);                                                         \
  }

#define GRAD_DEVICE_SWITCH(cpu, gpu, din, in, dout, l, h)                      \
  switch (din.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(din, in, dout, l, h);                                                  \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(din, in, dout, l, h);                                                  \
  }

#define GPU_ARGS()                                                             \
  const tcapint args[10U] {                                                    \
    a.offset, a.stride[0U], out.offset, out.stride[0U], 0U, 0U, 0U, 0U, 0U, 0U \
  }

#define GPU_GRAD_ARGS()                                                        \
  const tcapint args[10U] {                                                    \
    din.offset, din.stride[0U], in.offset, in.stride[0U], dout.offset,         \
        dout.stride[0U], 0U, 0U, 0U, 0U                                        \
  }

#define CPU_GRAD_KERNEL()                                                      \
  const auto fn = [&](const tcapint &i, const unsigned &) {                    \
    real1 ai = (*pi)[i];                                                       \
    if (ai > l && ai < h) {                                                    \
      pdi->add(i, (*po)[i]);                                                   \
    }                                                                          \
  }

namespace Weed {
void ClampKernel::cpu(const Tensor &a, const real1 &l, const real1 &h,
                      Tensor &out) {
  CPU_INIT_2(RealTensor, RealTensor);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    po->write(i, std::min(std::max((*pa)[i], l), h));
  };
  SPARSE_CPU_2_RUN(SparseCpuRealStorage);
}
void ClampKernel::cpu_grad_real(Tensor &din, const Tensor &in,
                                const Tensor &dout, const real1 &l,
                                const real1 &h) {
  CPU_GRAD_INIT_3(RealTensor, RealTensor, RealTensor);
  CPU_GRAD_KERNEL();
  SPARSE_CPU_GRAD_3_RUN(SparseCpuRealStorage, SparseCpuRealStorage);
}
void ClampKernel::cpu_grad_complex(Tensor &din, const Tensor &in,
                                   const Tensor &dout, const real1 &l,
                                   const real1 &h) {
  CPU_GRAD_INIT_3(ComplexTensor, RealTensor, ComplexTensor);
  CPU_GRAD_KERNEL();
  SPARSE_CPU_GRAD_3_RUN(SparseCpuComplexStorage, SparseCpuComplexStorage);
}
void ClampKernel::cpu_grad_mixed(Tensor &din, const Tensor &in,
                                 const Tensor &dout, const real1 &l,
                                 const real1 &h) {
  CPU_GRAD_INIT_3(ComplexTensor, RealTensor, RealTensor);
  CPU_GRAD_KERNEL();
  SPARSE_CPU_GRAD_3_RUN(SparseCpuComplexStorage, SparseCpuRealStorage);
}
#if ENABLE_GPU
void ClampKernel::gpu(const Tensor &a, const real1 &l, const real1 &h,
                      Tensor &out) {
  GPU_ARGS();
  GpuRealStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(a.storage);
  GpuRealStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(out.storage);
  const complex v = complex(l, h);
  a_storage->dev->RequestKernel(OCLAPI::OCL_API_CLAMP, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer}, 0U, &v);
}
void ClampKernel::gpu_grad_real(Tensor &din, const Tensor &in,
                                const Tensor &dout, const real1 &l,
                                const real1 &h) {
  GPU_GRAD(GpuRealStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_CLAMP_GRAD_REAL);
}
void ClampKernel::gpu_grad_complex(Tensor &din, const Tensor &in,
                                   const Tensor &dout, const real1 &l,
                                   const real1 &h) {
  GPU_GRAD(GpuComplexStorage, GpuRealStorage, GpuComplexStorage,
           OCL_API_CLAMP_GRAD_COMPLEX);
}
void ClampKernel::gpu_grad_mixed(Tensor &din, const Tensor &in,
                                 const Tensor &dout, const real1 &l,
                                 const real1 &h) {
  GPU_GRAD(GpuComplexStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_CLAMP_GRAD_MIXED);
}
#endif
void ClampKernel::clamp(const Tensor &a, const real1 &l, const real1 &h,
                        Tensor &out) {
  validate_all_same_device({&a, &out}, "ClampKernel::clamp");
  if ((a.storage->dtype != DType::REAL) ||
      (out.storage->dtype != DType::REAL)) {
    throw std::invalid_argument(
        "In Weed::clamp(a, l, h, out), arguments must all be real-number!");
  }
  const tcapint aSize = a.get_broadcast_size();
  const tcapint outSize = out.get_broadcast_size();
  if (aSize != outSize) {
    throw std::invalid_argument(
        "In Weed::clamp(a, l, h, out), out size does not match input size!");
  }
#if ENABLE_GPU
  DEVICE_SWITCH(cpu, gpu, a, l, h, out)
#else
  cpu(a, l, h, out);
#endif
}
void ClampKernel::clamp_grad(Tensor &din, const Tensor &in, const Tensor &dout,
                             const real1 &l, const real1 &h) {
  validate_all_same_device({&din, &in, &dout}, "ClampKernel::clamp_grad");
  if ((din.storage->dtype == DType::REAL) &&
      (dout.storage->dtype != DType::REAL)) {
    throw std::invalid_argument(
        "In Weed::clamp_grad(din, in, dout, l, h), dout dtype "
        "must upcast to dout dtype!");
  }
  const tcapint dinSize = din.get_broadcast_size();
  const tcapint inSize = in.get_broadcast_size();
  const tcapint doutSize = dout.get_broadcast_size();
  if ((dinSize != inSize) || (dinSize != doutSize)) {
    throw std::invalid_argument(
        "In Weed::clamp_grad(din, in, dout, l, h), sizes do not match!");
  }
  if (in.storage->dtype != DType::REAL) {
    throw std::invalid_argument("In Weed::clamp_grad(din, in, dout, l, h), "
                                "'in' dtype must be real-number!");
  }
  switch (din.storage->dtype) {
  case DType::COMPLEX:
    switch (dout.storage->dtype) {
    case DType::COMPLEX:
#if ENABLE_GPU
      GRAD_DEVICE_SWITCH(cpu_grad_complex, gpu_grad_complex, din, in, dout, l,
                         h);
#else
      cpu_grad_complex(din, in, dout, l, h);
#endif
      break;
    case DType::REAL:
    default:
#if ENABLE_GPU
      GRAD_DEVICE_SWITCH(cpu_grad_mixed, gpu_grad_mixed, din, in, dout, l, h);
#else
      cpu_grad_complex(din, in, dout, l, h);
#endif
    }
    break;
  case DType::REAL:
  default:
#if ENABLE_GPU
    GRAD_DEVICE_SWITCH(cpu_grad_real, gpu_grad_real, din, in, dout, l, h);
#else
    cpu_grad_real(din, in, dout, l, h);
#endif
  }
}

ClampKernel clamp_kernel;

void clamp(const Tensor &a, const real1 &l, const real1 &h, Tensor &out) {
  clamp_kernel.clamp(a, l, h, out);
}
void clamp_grad(Tensor &din, const Tensor &in, const Tensor &dout,
                const real1 &l, const real1 &h) {
  clamp_kernel.clamp_grad(din, in, dout, l, h);
}
} // namespace Weed
