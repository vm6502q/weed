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

#include "ops/div.hpp"
#include "common/parallel_for.hpp"
#include "ops/util.hpp"
#include "tensors/flat_tensors.hpp"

#define DIV_KERNEL()                                                           \
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    po->write(i, (*pa)[i] / (*pb)[i]);                                         \
  }

#define DISPATCH_GPU_KERNEL(type, type2, type3, api_call)                      \
  const tcapint args[10U]{                                                     \
      a.offset,       a.stride[0U], b.offset, b.stride[0U], out.offset,        \
      out.stride[0U], 0U,           0U,       0U,           0U};               \
  std::shared_ptr<type> a_storage =                                            \
      std::dynamic_pointer_cast<type>(a.storage);                              \
  std::shared_ptr<type2> b_storage =                                           \
      std::dynamic_pointer_cast<type2>(b.storage);                             \
  std::shared_ptr<type3> o_storage =                                           \
      std::dynamic_pointer_cast<type3>(out.storage);                           \
  a_storage->dev->RequestKernel(                                               \
      api_call, args, out.get_size(),                                          \
      {a_storage->buffer, b_storage->buffer, o_storage->buffer})

#define DEVICE_SWITCH(cpu, gpu)                                                \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, b, out);                                                            \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, b, out);                                                            \
  }

namespace Weed {
template <typename T1, typename T2, typename T3, typename T4>
static void cpu_div(const Tensor &a, const Tensor &b, Tensor &out) {
  CPU_INIT_3(T1, T2, T1);
  DIV_KERNEL();
  SPARSE_CPU_3_RUN(T3, T4);
}
static inline void cpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
  cpu_div<RealTensor, RealTensor, SparseCpuRealStorage, SparseCpuRealStorage>(
      a, b, out);
}
static inline void cpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
  cpu_div<ComplexTensor, ComplexTensor, SparseCpuComplexStorage,
          SparseCpuComplexStorage>(a, b, out);
}
static inline void cpu_mixed_c_left(const Tensor &a, const Tensor &b,
                                    Tensor &out) {
  cpu_div<ComplexTensor, RealTensor, SparseCpuComplexStorage,
          SparseCpuRealStorage>(a, b, out);
}
static inline void cpu_mixed_c_right(const Tensor &a, const Tensor &b,
                                     Tensor &out) {
  cpu_div<ComplexTensor, ComplexTensor, SparseCpuRealStorage,
          SparseCpuComplexStorage>(a, b, out);
}

#if ENABLE_GPU
static void gpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, GpuRealStorage, GpuRealStorage,
                      OCL_API_DIV_REAL);
}
static void gpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuComplexStorage, GpuComplexStorage,
                      OCL_API_DIV_COMPLEX);
}
static void gpu_mixed_c_left(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuRealStorage, GpuComplexStorage,
                      OCL_API_DIV_MIXED_C_LEFT);
}
static void gpu_mixed_c_right(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, GpuComplexStorage, GpuComplexStorage,
                      OCL_API_DIV_MIXED_C_RIGHT);
}
#endif

void div(const Tensor &a, const Tensor &b, Tensor &out) {
  validate_all_same_device({&a, &b, &out}, "DivKernel::div");
  const bool isAComplex = a.storage->dtype == DType::COMPLEX;
  const bool isBComplex = b.storage->dtype == DType::COMPLEX;
  const bool isOutComplex = out.storage->dtype == DType::COMPLEX;
  if (!isOutComplex && (isAComplex || isBComplex)) {
    throw std::invalid_argument(
        "Cannot combine complex tensors into real1 tensor!");
  }
  if (isOutComplex && (!isAComplex && !isBComplex)) {
    throw std::invalid_argument("Output tensor dtype mismatch!");
  }
  const tcapint aSize = a.get_broadcast_size();
  const tcapint bSize = b.get_broadcast_size();
  const tcapint outSize = out.get_broadcast_size();
  if (aSize != bSize) {
    throw std::invalid_argument(
        "In Weed::div(a, b, out), 'a' size does not match 'b' size!");
  }
  if (aSize != outSize) {
    throw std::invalid_argument(
        "In Weed::div(a, b, out), out size does not match input size!");
  }
  if (isAComplex && isBComplex) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_complex, gpu_complex);
#else
    cpu_complex(a, b, out);
#endif
  } else if (isAComplex) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_mixed_c_left, gpu_mixed_c_left);
#else
    cpu_mixed_c_left(a, b, out);
#endif
  } else if (isBComplex) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_mixed_c_right, gpu_mixed_c_right);
#else
    cpu_mixed_c_right(a, b, out);
#endif
  } else {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_real, gpu_real);
#else
    cpu_real(a, b, out);
#endif
  }
}
} // namespace Weed
