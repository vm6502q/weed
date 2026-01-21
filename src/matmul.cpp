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

#include "matmul.hpp"
#include "common/parallel_for.hpp"
#include "cpu_complex_storage.hpp"
#include "cpu_real_storage.hpp"
#include "gpu_complex_storage.hpp"
#include "gpu_real_storage.hpp"

#define CAST_STORAGE(out, in, type, ptr)                                       \
  type *out = static_cast<ptr *>(in.storage.get())->data.get() + in.offset

#define CPU_BY_TYPE(ltype, lstorage, rtype, rstorage, otype, ostorage, stype)  \
  if ((a.shape.size() != 2U) || (b.shape.size() != 2U) ||                      \
      (out.shape.size() != 2U)) {                                              \
    throw std::invalid_argument(                                               \
        "MatMul is only for matrices with 2 indices!");                        \
  }                                                                            \
  const vecCapIntGpu K = a.shape[1U];                                          \
  if (K != b.shape[0U]) {                                                      \
    throw std::invalid_argument(                                               \
        "MatMul operand dimensions aren't compatible!");                       \
  }                                                                            \
  const vecCapIntGpu M = a.shape[0U];                                          \
  const vecCapIntGpu N = b.shape[1U];                                          \
                                                                               \
  CAST_STORAGE(pa, a, ltype, lstorage);                                        \
  CAST_STORAGE(pb, b, rtype, rstorage);                                        \
  CAST_STORAGE(po, out, otype, ostorage);                                      \
                                                                               \
  const vecCapIntGpu I_a = a.stride[0U];                                       \
  const vecCapIntGpu I_b = b.stride[0U];                                       \
  const vecCapIntGpu I_o = out.stride[0U];                                     \
                                                                               \
  pfControl.par_for(0, M, [&](const vecCapIntGpu &i, const unsigned &cpu) {    \
    for (vecCapIntGpu j = 0; j < N; ++j) {                                     \
      stype sum = ZERO_R1;                                                     \
      for (vecCapIntGpu k = 0; k < K; ++k) {                                   \
        sum += pa[(i * K + k) * I_a] * pb[(k * N + j) * I_b];                  \
      }                                                                        \
      po[(i * N + j) * I_o] = sum;                                             \
    }                                                                          \
  })

#define _DEVICE_SWITCH(cpu, gpu)                                               \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, b, out);                                                            \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, b, out);                                                            \
  }

namespace Weed {
ParallelFor pfControl = ParallelFor();

struct matmul_kernel : MatMulKernel {
  void cpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
    CPU_BY_TYPE(real1, CpuRealStorage, real1, CpuRealStorage, real1,
                CpuRealStorage, real1);
  }
  void cpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
    CPU_BY_TYPE(complex, CpuComplexStorage, complex, CpuComplexStorage, complex,
                CpuComplexStorage, complex);
  }
  void cpu_mixed_c_left(const Tensor &a, const Tensor &b, Tensor &out) {
    CPU_BY_TYPE(complex, CpuComplexStorage, real1, CpuRealStorage, complex,
                CpuComplexStorage, complex);
  }
  void cpu_mixed_c_right(const Tensor &a, const Tensor &b, Tensor &out) {
    CPU_BY_TYPE(real1, CpuRealStorage, complex, CpuComplexStorage, complex,
                CpuComplexStorage, complex);
  }

  void gpu_real(const Tensor &a, const Tensor &b, Tensor &out) {}
  void gpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {}
  void gpu_mixed_c_left(const Tensor &a, const Tensor &b, Tensor &out) {}
  void gpu_mixed_c_right(const Tensor &a, const Tensor &b, Tensor &out) {}

  void matmul(const Tensor &a, const Tensor &b, Tensor &out) {
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
    if (isAComplex && isBComplex) {
      _DEVICE_SWITCH(cpu_complex, gpu_complex);
    } else if (isAComplex) {
      _DEVICE_SWITCH(cpu_mixed_c_left, gpu_mixed_c_left);
    } else if (isBComplex) {
      _DEVICE_SWITCH(cpu_mixed_c_right, gpu_mixed_c_right);
    } else {
      _DEVICE_SWITCH(cpu_real, gpu_real);
    }
  }
};
} // namespace Weed
