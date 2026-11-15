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

#include "ops/sub.hpp"
#include "common/parallel_for.hpp"
#include "ops/util.hpp"
#include "tensors/flat_tensors.hpp"

#if ENABLE_GPU
#include "ops/gpu_flat.hpp"
#endif

#define SUB_KERNEL()                                                           \
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    po->write(i, (*pa)[i] - (*pb)[i]);                                         \
  }

#define DISPATCH_GPU_KERNEL(type, type2, type3, api_call)                      \
  GPU_FLAT_BINARY(type, type2, type3, api_call)

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
template <typename T1, typename T2, typename T3, typename T4, typename T5>
static void cpu(const Tensor &a, const Tensor &b, Tensor &out) {
  CPU_INIT_3(T1, T2, T3);
  SUB_KERNEL();
  SPARSE_CPU_3_RUN(T4, T5);
}
static inline void cpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
  cpu<RealTensor, RealTensor, RealTensor, SparseCpuRealStorage,
      SparseCpuRealStorage>(a, b, out);
}
static inline void cpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
  cpu<ComplexTensor, ComplexTensor, ComplexTensor, SparseCpuComplexStorage,
      SparseCpuComplexStorage>(a, b, out);
}
static inline void cpu_mixed_c_left(const Tensor &a, const Tensor &b,
                                    Tensor &out) {
  cpu<ComplexTensor, RealTensor, ComplexTensor, SparseCpuComplexStorage,
      SparseCpuRealStorage>(a, b, out);
}
static inline void cpu_mixed_c_right(const Tensor &a, const Tensor &b,
                                     Tensor &out) {
  cpu<RealTensor, ComplexTensor, ComplexTensor, SparseCpuRealStorage,
      SparseCpuRealStorage>(a, b, out);
}

#if ENABLE_GPU
static void gpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, GpuRealStorage, GpuRealStorage,
                      OCL_API_SUB_REAL);
}
static void gpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuComplexStorage, GpuComplexStorage,
                      OCL_API_SUB_COMPLEX);
}
static void gpu_mixed_c_left(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuRealStorage, GpuComplexStorage,
                      OCL_API_SUB_MIXED_C_LEFT);
}
static void gpu_mixed_c_right(const Tensor &a, const Tensor &b, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, GpuComplexStorage, GpuComplexStorage,
                      OCL_API_SUB_MIXED_C_RIGHT);
}
#endif

void sub(const Tensor &a, const Tensor &b, Tensor &out) {
  validate_all_same_device({&a, &b, &out}, "SubKernel::sub");
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
        "In Weed::sub(a, b, out), 'a' size does not match 'b' size!");
  }
  if (aSize != outSize) {
    throw std::invalid_argument(
        "In Weed::sub(a, b, out), out size does not match input size!");
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
