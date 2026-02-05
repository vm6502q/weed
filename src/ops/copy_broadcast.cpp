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

#include "ops/copy_broadcast.hpp"
#include "common/parallel_for.hpp"
#include "ops/util.hpp"
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

#define COPY_KERNEL()                                                          \
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    pa->write(i, (*pb)[i]);                                                    \
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
template <typename T1, typename T2> static void cpu_real(Tensor &a, const Tensor &b) {
  CPU_INIT_2_IN_PLACE(T1, T1);
  COPY_KERNEL();
  SPARSE_CPU_2_SWITCH(T2);
}
static inline void cpu_real(Tensor &a, const Tensor &b) {
  cpu_real<RealTensor, SparseCpuRealStorage>(a, b);
}
static inline void cpu_complex(Tensor &a, const Tensor &b) {
  cpu_real<ComplexTensor, SparseCpuComplexStorage>(a, b);
}
#if ENABLE_GPU
static inline void gpu_real(Tensor &a, const Tensor &b) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, GpuRealStorage, OCL_API_COPY_REAL);
}
static inline void gpu_complex(Tensor &a, const Tensor &b) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuComplexStorage,
                      OCL_API_COPY_COMPLEX);
}
#endif

void copy_broadcast(Tensor &a, const Tensor &b) {
  validate_all_same_device({&a, &b}, "CopyKernel::copy_broadcast");
  const tcapint aSize = a.get_size();
  const tcapint bSize = b.get_broadcast_size();
  if (aSize != bSize) {
    throw std::invalid_argument("In CopyKernel::copy_broadcast(a, b), 'a' "
                                "size does not match 'b' size!");
  }
  const bool isAComplex = a.storage->dtype == DType::COMPLEX;
  const bool isBComplex = b.storage->dtype == DType::COMPLEX;
  if (isBComplex != isAComplex) {
    throw std::invalid_argument(
        "Cannot combine complex tensors into real1 tensor!");
  }
  if (isAComplex) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_complex, gpu_complex, a, b);
#else
    cpu_complex(a, b);
#endif
  } else {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_real, gpu_real, a, b);
#else
    cpu_real(a, b);
#endif
  }
}
} // namespace Weed
