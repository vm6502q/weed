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

#include "in_place.hpp"
#include "common/parallel_for.hpp"
#include "cpu_complex_storage.hpp"
#include "cpu_real_storage.hpp"
#include "gpu_complex_storage.hpp"
#include "gpu_real_storage.hpp"

#define CAST_STORAGE(out, in, type, ptr)                                       \
  type *out = static_cast<ptr *>(in.storage.get())->data.get() + in.offset

#define DEVICE_SWITCH(cpu, gpu, a, b)                                          \
  switch (a.storage->device) {                                                 \
  case DeviceTag::GPU:                                                         \
    gpu(a, b);                                                                 \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, b);                                                                 \
  }

#define ADD_KERNEL()                                                           \
  const vecCapIntGpu I_a = (vecCapIntGpu)a.stride[0U];                         \
  const vecCapIntGpu I_b = (vecCapIntGpu)b.stride[0U];                         \
  const size_t n = a.get_size();                                               \
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {    \
    pa[i * I_a] += pb[i * I_b];                                                \
  })

#define SUB_KERNEL()                                                           \
  const vecCapIntGpu I_a = (vecCapIntGpu)a.stride[0U];                         \
  const vecCapIntGpu I_b = (vecCapIntGpu)b.stride[0U];                         \
  const size_t n = a.get_size();                                               \
  pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {    \
    pa[i * I_a] -= pb[i * I_b];                                                \
  })

#define DISPATCH_GPU_KERNEL(type, type2, api_call)                             \
  const vecCapIntGpu args[10U]{(vecCapIntGpu)(a.offset),                       \
                               (vecCapIntGpu)(a.stride[0U]),                   \
                               (vecCapIntGpu)(b.offset),                       \
                               (vecCapIntGpu)(b.stride[0U]),                   \
                               0U,                                             \
                               0U,                                             \
                               0U,                                             \
                               0U,                                             \
                               0U,                                             \
                               0U};                                            \
  std::shared_ptr<type> a_storage =                                            \
      std::dynamic_pointer_cast<type>(a.storage);                              \
  std::shared_ptr<type2> b_storage =                                           \
      std::dynamic_pointer_cast<type2>(b.storage);                             \
  a_storage->gpu->RequestKernel(api_call, args, a.get_size(),                  \
                                {a_storage->buffer, b_storage->buffer})

namespace Weed {
static void cpu_real_add(Tensor &a, const Tensor &b) {
  CAST_STORAGE(pa, a, real1, CpuRealStorage);
  CAST_STORAGE(pb, b, real1, CpuRealStorage);

  ADD_KERNEL();
}
static void cpu_complex_add(Tensor &a, const Tensor &b) {
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(pb, b, complex, CpuComplexStorage);

  ADD_KERNEL();
}
static void cpu_mixed_add(Tensor &a, const Tensor &b) {
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(pb, b, real1, CpuRealStorage);

  ADD_KERNEL();
}
static void gpu_real_add(Tensor &a, const Tensor &b) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, GpuRealStorage,
                      OCL_API_ADD_IN_PLACE_REAL);
}
static void gpu_complex_add(Tensor &a, const Tensor &b) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuComplexStorage,
                      OCL_API_ADD_IN_PLACE_COMPLEX);
}
static void gpu_mixed_add(Tensor &a, const Tensor &b) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuRealStorage,
                      OCL_API_ADD_IN_PLACE_MIXED);
}

static void cpu_real_sub(Tensor &a, const Tensor &b) {
  CAST_STORAGE(pa, a, real1, CpuRealStorage);
  CAST_STORAGE(pb, b, real1, CpuRealStorage);

  SUB_KERNEL();
}
static void cpu_complex_sub(Tensor &a, const Tensor &b) {
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(pb, b, complex, CpuComplexStorage);

  SUB_KERNEL();
}
static void cpu_mixed_sub(Tensor &a, const Tensor &b) {
  CAST_STORAGE(pa, a, complex, CpuComplexStorage);
  CAST_STORAGE(pb, b, real1, CpuRealStorage);

  SUB_KERNEL();
}
static void gpu_real_sub(Tensor &a, const Tensor &b) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, GpuRealStorage,
                      OCL_API_SUB_IN_PLACE_REAL);
}
static void gpu_complex_sub(Tensor &a, const Tensor &b) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuComplexStorage,
                      OCL_API_SUB_IN_PLACE_COMPLEX);
}
static void gpu_mixed_sub(Tensor &a, const Tensor &b) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuRealStorage,
                      OCL_API_SUB_IN_PLACE_MIXED);
}

void InPlaceKernel::in_place(Tensor &a, const Tensor &b) {
  const bool isAComplex = a.storage->dtype == DType::COMPLEX;
  const bool isBComplex = b.storage->dtype == DType::COMPLEX;
  if (isBComplex && !isAComplex) {
    throw std::invalid_argument(
        "Cannot combine complex tensors into real1 tensor!");
  }
  if (isAComplex && isBComplex) {
    DEVICE_SWITCH(cpu_complex, gpu_complex, a, b);
  } else if (isAComplex) {
    DEVICE_SWITCH(cpu_mixed, gpu_mixed, a, b);
  } else {
    DEVICE_SWITCH(cpu_real, gpu_real, a, b);
  }
}

InPlaceKernel add_in_place_kernel = {cpu_real_add,    gpu_real_add,
                                     cpu_complex_add, gpu_complex_add,
                                     cpu_mixed_add,   gpu_mixed_add};
InPlaceKernel sub_in_place_kernel = {cpu_real_sub,    gpu_real_sub,
                                     cpu_complex_sub, gpu_complex_sub,
                                     cpu_mixed_sub,   gpu_mixed_sub};

void add_in_place(Tensor &a, const Tensor &b) {
  add_in_place_kernel.in_place(a, b);
}
void sub_in_place(Tensor &a, const Tensor &b) {
  sub_in_place_kernel.in_place(a, b);
}
} // namespace Weed
