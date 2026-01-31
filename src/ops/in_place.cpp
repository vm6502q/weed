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

#include "ops/in_place.hpp"
#include "common/parallel_for.hpp"
#include "tensors/flat_tensors.hpp"

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
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    pa->add(i, (*pb)[i]);                                                      \
  }

#define SUB_KERNEL()                                                           \
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    pa->add(i, -(*pb)[i]);                                                     \
  }

#define DISPATCH_GPU_KERNEL(type, type2, api_call)                             \
  const tcapint args[10U]{a.offset, a.stride[0U], b.offset, b.stride[0U], 0U,  \
                          0U,       0U,           0U,       0U,           0U}; \
  std::shared_ptr<type> a_storage =                                            \
      std::dynamic_pointer_cast<type>(a.storage);                              \
  std::shared_ptr<type2> b_storage =                                           \
      std::dynamic_pointer_cast<type2>(b.storage);                             \
  a_storage->dev->RequestKernel(api_call, args, a.get_size(),                  \
                                {a_storage->buffer, b_storage->buffer})

namespace Weed {
static void cpu_real_add(Tensor &a, const Tensor &b) {
  CPU_INIT_2_IN_PLACE(RealTensor, RealTensor);
  ADD_KERNEL();
  SPARSE_CPU_2_SWITCH(SparseCpuRealStorage, SparseCpuRealStorage);
}
static void cpu_complex_add(Tensor &a, const Tensor &b) {
  CPU_INIT_2_IN_PLACE(ComplexTensor, ComplexTensor);
  ADD_KERNEL();
  SPARSE_CPU_2_SWITCH(SparseCpuComplexStorage, SparseCpuComplexStorage);
}
static void cpu_mixed_add(Tensor &a, const Tensor &b) {
  CPU_INIT_2_IN_PLACE(ComplexTensor, RealTensor);
  ADD_KERNEL();
  SPARSE_CPU_2_SWITCH(SparseCpuComplexStorage, SparseCpuRealStorage);
}
#if ENABLE_GPU
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
#endif

static void cpu_real_sub(Tensor &a, const Tensor &b) {
  CPU_INIT_2_IN_PLACE(RealTensor, RealTensor);
  SUB_KERNEL();
  SPARSE_CPU_2_SWITCH(SparseCpuRealStorage, SparseCpuRealStorage);
}
static void cpu_complex_sub(Tensor &a, const Tensor &b) {
  CPU_INIT_2_IN_PLACE(ComplexTensor, ComplexTensor);
  SUB_KERNEL();
  SPARSE_CPU_2_SWITCH(SparseCpuComplexStorage, SparseCpuComplexStorage);
}
static void cpu_mixed_sub(Tensor &a, const Tensor &b) {
  CPU_INIT_2_IN_PLACE(ComplexTensor, RealTensor);
  SUB_KERNEL();
  SPARSE_CPU_2_SWITCH(SparseCpuComplexStorage, SparseCpuRealStorage);
}

#if ENABLE_GPU
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
#endif

void InPlaceKernel::in_place(Tensor &a, const Tensor &b) {
  const bool isAComplex = a.storage->dtype == DType::COMPLEX;
  const bool isBComplex = b.storage->dtype == DType::COMPLEX;
  if (isBComplex && !isAComplex) {
    throw std::invalid_argument(
        "Cannot combine complex tensors into real1 tensor!");
  }
  if (isAComplex && isBComplex) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_complex, gpu_complex, a, b);
#else
    cpu_complex(a, b);
#endif
  } else if (isAComplex) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_mixed, gpu_mixed, a, b);
#else
    cpu_mixed(a, b);
#endif
  } else {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_real, gpu_real, a, b);
#else
    cpu_real(a, b);
#endif
  }
}

InPlaceKernel add_in_place_kernel = {cpu_real_add,  cpu_complex_add,
                                     cpu_mixed_add,
#if ENABLE_GPU
                                     gpu_real_add,  gpu_complex_add,
                                     gpu_mixed_add
#endif
};
InPlaceKernel sub_in_place_kernel = {cpu_real_sub,  cpu_complex_sub,
                                     cpu_mixed_sub,
#if ENABLE_GPU
                                     gpu_real_sub,  gpu_complex_sub,
                                     gpu_mixed_sub
#endif
};

void add_in_place(Tensor &a, const Tensor &b) {
  add_in_place_kernel.in_place(a, b);
}
void sub_in_place(Tensor &a, const Tensor &b) {
  sub_in_place_kernel.in_place(a, b);
}
} // namespace Weed
