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

#include "ops/matmul.hpp"
#include "common/parallel_for.hpp"
#include "storage/all_storage.hpp"

#define CPU_BY_TYPE(ltype, lstorage, rtype, rstorage, otype, ostorage, stype)  \
  MatrixDim d = get_dim(a, b, out);                                            \
                                                                               \
  CAST_STORAGE(pa, a, ltype, lstorage);                                        \
  CAST_STORAGE(pb, b, rtype, rstorage);                                        \
  CAST_STORAGE(po, out, otype, ostorage);                                      \
                                                                               \
  pfControl.par_for(0, (d.M) * (d.N),                                          \
                    [&](const vecCapIntGpu &l, const unsigned &cpu) {          \
                      const vecCapIntGpu i = l / d.N;                          \
                      const vecCapIntGpu j = l % d.N;                          \
                      stype sum = ZERO_R1;                                     \
                      for (vecCapIntGpu k = 0; k < d.K; ++k) {                 \
                        const auto a_idx = (d.A_o + i * d.A_s0 + k * d.A_s1);  \
                        const auto b_idx = (d.B_o + k * d.B_s0 + j * d.B_s1);  \
                        sum += pa[a_idx] * pb[b_idx];                          \
                      }                                                        \
                      const auto o_idx = (d.O_o + i * d.O_s0 + j * d.O_s1);    \
                      po[o_idx] = sum;                                         \
                    })

#define GPU_BY_TYPE(ltype, lstorage, rtype, rstorage, otype, ostorage, call)   \
  MatrixDim d = get_dim(a, b, out);                                            \
  const vecCapIntGpu args[10U]{d.A_o,  d.A_s0, d.B_o,  d.B_s0, d.O_o,          \
                               d.O_s0, d.A_s1, d.B_s1, d.O_s1, d.K};           \
  lstorage a_storage = std::dynamic_pointer_cast<ltype>(a.storage);            \
  rstorage b_storage = std::dynamic_pointer_cast<rtype>(b.storage);            \
  ostorage o_storage = std::dynamic_pointer_cast<otype>(out.storage);          \
  a_storage->gpu->RequestKernel(                                               \
      OCLAPI::call, args, d.M,                                                 \
      {a_storage->buffer, b_storage->buffer, o_storage->buffer}, d.N)

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
struct MatrixDim {
  vecCapIntGpu M, K, N;
  vecCapIntGpu A_o, B_o, O_o;
  vecCapIntGpu A_s0, A_s1;
  vecCapIntGpu B_s0, B_s1;
  vecCapIntGpu O_s0, O_s1;
};

MatrixDim MatMulKernel::get_dim(const Tensor &a, const Tensor &b, Tensor &out) {
  if ((a.shape.size() != 2U) || (b.shape.size() != 2U) ||
      (out.shape.size() != 2U)) {
    throw std::invalid_argument("MatMul is only for matrices with 2 indices!");
  }
  MatrixDim d;
  d.K = a.shape[1U];
  if (d.K != b.shape[0U]) {
    throw std::invalid_argument("MatMul operand dimensions aren't compatible!");
  }
  d.M = a.shape[0U];
  d.N = b.shape[1U];
  if ((d.M != out.shape[0U]) || (d.N != out.shape[1U])) {
    throw std::invalid_argument("MatMul output dimensions don't match inputs!");
  }

  d.A_o = a.offset;
  d.B_o = b.offset;
  d.O_o = out.offset;
  d.A_s0 = a.stride[0];
  d.A_s1 = a.stride[1];
  d.B_s0 = b.stride[0];
  d.B_s1 = b.stride[1];
  d.O_s0 = out.stride[0];
  d.O_s1 = out.stride[1];

  return d;
}

void MatMulKernel::cpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
  CPU_BY_TYPE(real1, CpuRealStorage, real1, CpuRealStorage, real1,
              CpuRealStorage, real1);
}
void MatMulKernel::cpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
  CPU_BY_TYPE(complex, CpuComplexStorage, complex, CpuComplexStorage, complex,
              CpuComplexStorage, complex);
}
void MatMulKernel::cpu_mixed_c_left(const Tensor &a, const Tensor &b,
                                    Tensor &out) {
  CPU_BY_TYPE(complex, CpuComplexStorage, real1, CpuRealStorage, complex,
              CpuComplexStorage, complex);
}
void MatMulKernel::cpu_mixed_c_right(const Tensor &a, const Tensor &b,
                                     Tensor &out) {
  CPU_BY_TYPE(real1, CpuRealStorage, complex, CpuComplexStorage, complex,
              CpuComplexStorage, complex);
}

#if ENABLE_GPU
void MatMulKernel::gpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
  GPU_BY_TYPE(GpuRealStorage, GpuRealStoragePtr, GpuRealStorage,
              GpuRealStoragePtr, GpuRealStorage, GpuRealStoragePtr,
              OCL_API_MATMUL_REAL);
}
void MatMulKernel::gpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
  GPU_BY_TYPE(GpuComplexStorage, GpuComplexStoragePtr, GpuComplexStorage,
              GpuComplexStoragePtr, GpuComplexStorage, GpuComplexStoragePtr,
              OCL_API_MATMUL_COMPLEX);
}
void MatMulKernel::gpu_mixed_c_left(const Tensor &a, const Tensor &b,
                                    Tensor &out) {
  GPU_BY_TYPE(GpuComplexStorage, GpuComplexStoragePtr, GpuRealStorage,
              GpuRealStoragePtr, GpuComplexStorage, GpuComplexStoragePtr,
              OCL_API_MATMUL_MIXED_C_LEFT);
}
void MatMulKernel::gpu_mixed_c_right(const Tensor &a, const Tensor &b,
                                     Tensor &out) {
  GPU_BY_TYPE(GpuRealStorage, GpuRealStoragePtr, GpuComplexStorage,
              GpuComplexStoragePtr, GpuComplexStorage, GpuComplexStoragePtr,
              OCL_API_MATMUL_MIXED_C_RIGHT);
}
#endif

void MatMulKernel::matmul(const Tensor &a, const Tensor &b, Tensor &out) {
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

MatMulKernel matmul_kernel;

void matmul(const Tensor &a, const Tensor &b, Tensor &out) {
  matmul_kernel.matmul(a, b, out);
}
} // namespace Weed
