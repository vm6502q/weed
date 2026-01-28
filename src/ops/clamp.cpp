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
#include "storage/all_storage.hpp"

#define GPU_GRAD(type1, type2, type3, api_call)                                \
  GPU_GRAD_ARGS();                                                             \
  std::shared_ptr<type1> dy_storage =                                          \
      std::dynamic_pointer_cast<type1>(dy.storage);                            \
  std::shared_ptr<type2> dx_storage =                                          \
      std::dynamic_pointer_cast<type2>(dx.storage);                            \
  std::shared_ptr<type3> x_storage =                                           \
      std::dynamic_pointer_cast<type3>(x.storage);                             \
  const complex v = complex(l, h);                                             \
  x_storage->dev->RequestKernel(                                               \
      OCLAPI::api_call, args, x.get_size(),                                    \
      {dy_storage->buffer, x_storage->buffer, dx_storage->buffer}, 0U, &v)

#define DEVICE_SWITCH(cpu, gpu, a, l, h, out)                                  \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, l, h, out);                                                         \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, l, h, out);                                                         \
  }

#define GRAD_DEVICE_SWITCH(cpu, gpu, dy, x, l, h, dx)                          \
  switch (dy.storage->device) {                                                \
  case DeviceTag::GPU:                                                         \
    gpu(dy, x, l, h, dx);                                                      \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(dy, x, l, h, dx);                                                      \
  }

#define GPU_ARGS()                                                             \
  const tcapint args[10U] {                                                    \
    a.offset, a.stride[0U], out.offset, out.stride[0U], 0U, 0U, 0U, 0U, 0U, 0U \
  }

#define GPU_GRAD_ARGS()                                                        \
  const tcapint args[10U] {                                                    \
    dy.offset, dy.stride[0U], x.offset, x.stride[0U], dx.offset,               \
        dx.stride[0U], 0U, 0U, 0U, 0U                                          \
  }

#define CPU_GRAD_KERNEL()                                                      \
  pfControl.par_for(0, n, [&](const tcapint &i, const unsigned &) {            \
    real1 ai = (*pi)[O_i + i * I_i];                                           \
    if (ai > l && ai < h) {                                                    \
      pdi->add(O_d + i * I_d, (*po)[O_o + i * I_o]);                           \
    }                                                                          \
  })

namespace Weed {
void ClampKernel::cpu(const Tensor &a, const real1 &l, const real1 &h,
                      Tensor &out) {
  CPU_INIT_2(RealStorage, RealStorage);
  pfControl.par_for(0, n, [&](const tcapint &i, const unsigned &cpu) {
    po->write(i * I_o, std::min(std::max((*pa)[O_a + i * I_a], l), h));
  });
}
void ClampKernel::cpu_grad_real(const Tensor &dout, const Tensor &in,
                                const real1 &l, const real1 &h, Tensor &din) {
  CPU_GRAD_INIT_3(RealStorage, RealStorage, RealStorage);
  CPU_GRAD_KERNEL();
}
void ClampKernel::cpu_grad_complex(const Tensor &dout, const Tensor &in,
                                   const real1 &l, const real1 &h,
                                   Tensor &din) {
  CPU_GRAD_INIT_3(ComplexStorage, RealStorage, ComplexStorage);
  CPU_GRAD_KERNEL();
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
void ClampKernel::gpu_grad_real(const Tensor &dy, const Tensor &x,
                                const real1 &l, const real1 &h, Tensor &dx) {
  GPU_GRAD(GpuRealStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_CLAMP_GRAD_REAL);
}
void ClampKernel::gpu_grad_complex(const Tensor &dy, const Tensor &x,
                                   const real1 &l, const real1 &h, Tensor &dx) {
  GPU_GRAD(GpuComplexStorage, GpuComplexStorage, GpuRealStorage,
           OCL_API_CLAMP_GRAD_COMPLEX);
}
#endif
void ClampKernel::clamp(const Tensor &a, const real1 &l, const real1 &h,
                        Tensor &out) {
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
  cpu_real(a, l, h, out);
#endif
}
void ClampKernel::clamp_grad(const Tensor &dy, const Tensor &x, const real1 &l,
                             const real1 &h, Tensor &dx) {
  if (x.storage->dtype != DType::REAL) {
    throw std::invalid_argument(
        "In Weed::clamp_grad(dy, x, l, h, dx), x must be real-number!");
  }
  if (dy.storage->dtype != dy.storage->dtype) {
    throw std::invalid_argument(
        "In Weed::clamp_grad(dy, x, l, h, dx), dy dtype must match dx dtype!");
  }
  switch (dy.storage->dtype) {
  case DType::COMPLEX:
#if ENABLE_GPU
    GRAD_DEVICE_SWITCH(cpu_grad_complex, gpu_grad_complex, dy, x, l, h, dx)
#else
    cpu_grad_complex(dy, x, l, h, dx);
#endif
    break;
  case DType::REAL:
  default:
#if ENABLE_GPU
    GRAD_DEVICE_SWITCH(cpu_grad_real, gpu_grad_real, dy, x, l, h, dx)
#else
    cpu_grad_real(dy, x, l, h, dx);
#endif
  }
}

ClampKernel clamp_kernel;

void clamp(const Tensor &a, const real1 &l, const real1 &h, Tensor &out) {
  clamp_kernel.clamp(a, l, h, out);
}
void clamp_grad(const Tensor &dy, const Tensor &x, const real1 &l,
                const real1 &h, Tensor &dx) {
  clamp_kernel.clamp_grad(dy, x, l, h, dx);
}
} // namespace Weed
