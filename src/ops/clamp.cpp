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
  const vecCapIntGpu args[10U] {                                               \
    (vecCapIntGpu)(a.offset), (vecCapIntGpu)(a.stride[0U]),                    \
        (vecCapIntGpu)(out.offset), (vecCapIntGpu)(out.stride[0U]), 0U, 0U,    \
        0U, 0U, 0U, 0U                                                         \
  }

#define GPU_GRAD_ARGS()                                                        \
  const vecCapIntGpu args[10U] {                                               \
    (vecCapIntGpu)(dy.offset), (vecCapIntGpu)(dy.stride[0U]),                  \
        (vecCapIntGpu)(x.offset), (vecCapIntGpu)(x.stride[0U]),                \
        (vecCapIntGpu)(dx.offset), (vecCapIntGpu)(dx.stride[0U]), 0U, 0U, 0U,  \
        0U                                                                     \
  }

#define CPU_GRAD(type, storage)                                                \
  CAST_STORAGE(pdx, dx, type, storage);                                        \
  CAST_STORAGE(px, x, real1, CpuRealStorage);                                  \
  CAST_STORAGE(pdy, dy, type, storage);                                        \
                                                                               \
  const vecCapIntGpu I_dx = dx.stride[0];                                      \
  const vecCapIntGpu I_x = x.stride[0];                                        \
  const vecCapIntGpu I_dy = dy.stride[0];                                      \
  const size_t n = x.get_size();                                               \
                                                                               \
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &) {       \
    real1 xi = px[i * I_x];                                                    \
    if (xi > l && xi < h) {                                                    \
      pdx[i * I_dx] += pdy[i * I_dy];                                          \
    }                                                                          \
  })

namespace Weed {
void ClampKernel::cpu(const Tensor &a, const real1 &l, const real1 &h,
                      Tensor &out) {
  const vecCapIntGpu I_a = (vecCapIntGpu)a.stride[0U];
  const vecCapIntGpu I_o = (vecCapIntGpu)out.stride[0U];
  CAST_STORAGE(pa, a, real1, CpuRealStorage);
  CAST_STORAGE(po, out, real1, CpuRealStorage);
  size_t n = out.storage->size;
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
    po[i * I_o] = std::min(std::max(pa[i * I_a], l), h);
  });
}
void ClampKernel::cpu_grad_real(const Tensor &dy, const Tensor &x,
                                const real1 &l, const real1 &h, Tensor &dx) {
  CPU_GRAD(real1, CpuRealStorage);
}
void ClampKernel::cpu_grad_complex(const Tensor &dy, const Tensor &x,
                                   const real1 &l, const real1 &h, Tensor &dx) {
  CPU_GRAD(complex, CpuComplexStorage);
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
  const size_t aSize = a.get_broadcast_size();
  const size_t outSize = out.get_broadcast_size();
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
