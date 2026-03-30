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
#include "ops/util.hpp"
#include "tensors/flat_tensors.hpp"

#if WEED_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#define CPU_HEADER(storage1, storage2, storage3)                               \
  MatrixDim d = get_dim(a, b, out);                                            \
                                                                               \
  GET_STORAGE(storage1, a, pa);                                                \
  GET_STORAGE(storage2, b, pb);                                                \
  GET_STORAGE(storage3, out, po);

#define CPU_BY_TYPE(stype)                                                     \
  pfControl.par_for(0, (d.M) * (d.N),                                          \
                    [&](const tcapint &l, const unsigned &cpu) {               \
                      const tcapint i = l / d.N;                               \
                      const tcapint j = l % d.N;                               \
                      stype sum = ZERO_R1;                                     \
                      for (tcapint k = 0; k < d.K; ++k) {                      \
                        const auto a_idx = d.A_o + i * d.A_s0 + k * d.A_s1;    \
                        const auto b_idx = d.B_o + k * d.B_s0 + j * d.B_s1;    \
                        sum += (*pa)[a_idx] * (*pb)[b_idx];                    \
                      }                                                        \
                      const auto o_idx = d.O_o + i * d.O_s0 + j * d.O_s1;      \
                      po->write(o_idx, sum);                                   \
                    })

#define TILE_BY_TYPE(ltype, lstorage, rtype, rstorage, otype, ostorage, call)  \
  MatrixDim d = get_dim(a, b, out);                                            \
  const tcapint args[12U]{d.A_o,  d.A_s0, d.B_o,  d.B_s0, d.O_o, d.O_s0,       \
                          d.A_s1, d.B_s1, d.O_s1, d.K,    d.M,   d.N};         \
  lstorage a_storage = std::dynamic_pointer_cast<ltype>(a.storage);            \
  rstorage b_storage = std::dynamic_pointer_cast<rtype>(b.storage);            \
  ostorage o_storage = std::dynamic_pointer_cast<otype>(out.storage);          \
  {                                                                            \
    EventVecPtr waitVec = a_storage->dev->ResetWaitEvents();                   \
    PoolItemPtr poolItem = a_storage->dev->GetFreePoolItem();                  \
    cl::Event writeArgsEvent;                                                  \
    a_storage->dev->tryOcl("Failed to write matmul args", [&] {                \
      return a_storage->dev->queue.enqueueWriteBuffer(                         \
          *(poolItem->vciBuffer), CL_FALSE, 0U, sizeof(tcapint) * 12U, args,   \
          waitVec.get(), &writeArgsEvent);                                     \
    });                                                                        \
    writeArgsEvent.wait();                                                     \
    const size_t gws_m =                                                       \
        ((d.M + WEED_TILE_SIZE - 1U) / WEED_TILE_SIZE) * WEED_TILE_SIZE;       \
    const size_t gws_n =                                                       \
        ((d.N + WEED_TILE_SIZE - 1U) / WEED_TILE_SIZE) * WEED_TILE_SIZE;       \
    a_storage->dev->QueueCall(OCLAPI::call, gws_m, WEED_TILE_SIZE,             \
                              {a_storage->buffer, b_storage->buffer,           \
                               o_storage->buffer, poolItem->vciBuffer},        \
                              gws_n, WEED_TILE_SIZE);                          \
  }

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
  tcapint M, K, N;
  tcapint A_o, B_o, O_o;
  tcapint A_s0, A_s1;
  tcapint B_s0, B_s1;
  tcapint O_s0, O_s1;
};

static MatrixDim get_dim(const Tensor &a, const Tensor &b, Tensor &out) {
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
  if ((d.M != out.shape[0U]) || (d.N != out.shape[1])) {
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

template <typename T1, typename T2, typename T3, typename T4>
static void cpu(const Tensor &a, const Tensor &b, Tensor &out) {
  CPU_HEADER(T1, T2, T3);
  CPU_BY_TYPE(T4);
}
static inline void cpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
#if WEED_BLAS && (WEED_FPPOW > 4) && (WEED_FPPOW < 7)
  MatrixDim d = get_dim(a, b, out);

  // BLAS sgemm requires contiguous storage with unit or leading-dimension
  // strides. Check if we can use BLAS directly.
  // Column-major: A is M×K, B is K×N, C is M×N.
  // BLAS leading dimensions: lda=stride of column = stride[1] for col-major.
  // Weed col-major: stride[0]=1 (row stride), stride[1]=M (col stride).
  // So lda = d.A_s1, ldb = d.B_s1, ldc = d.O_s1.
  // This works as long as offsets are zero and strides are contiguous.

  const bool a_contiguous = (d.A_o == 0U) && (d.A_s0 == 1U);
  const bool b_contiguous = (d.B_o == 0U) && (d.B_s0 == 1U);
  const bool o_contiguous = (d.O_o == 0U) && (d.O_s0 == 1U);

  if (a_contiguous && b_contiguous && o_contiguous) {
    // Get raw pointers to storage
    auto *a_store = static_cast<CpuRealStorage *>(a.storage.get());
    auto *b_store = static_cast<CpuRealStorage *>(b.storage.get());
    auto *o_store = static_cast<CpuRealStorage *>(out.storage.get());

#if WEED_FPPOW == 5
    cblas_sgemm(
#else
    cblas_dgemm(
#endif
        CblasColMajor, CblasNoTrans, CblasNoTrans,
        (blasint)d.M,                         // rows of A and C
        (blasint)d.N,                         // cols of B and C
        (blasint)d.K,                         // cols of A, rows of B
        ONE_R1_F,                             // alpha
        a_store->data.get(), (blasint)d.A_s1, // A, lda
        b_store->data.get(), (blasint)d.B_s1, // B, ldb
        ZERO_R1_F,                            // beta (overwrite C)
        o_store->data.get(), (blasint)d.O_s1  // C, ldc
    );
    return;
  }
  // Fall through to hand-rolled for non-contiguous cases
#endif
  cpu<RealStorage, RealStorage, RealStorage, real1>(a, b, out);
}
static inline void cpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
#if WEED_BLAS && (WEED_FPPOW > 4) && (WEED_FPPOW < 7)
  MatrixDim d = get_dim(a, b, out);

  const bool a_contiguous = (d.A_o == 0U) && (d.A_s0 == 1U);
  const bool b_contiguous = (d.B_o == 0U) && (d.B_s0 == 1U);
  const bool o_contiguous = (d.O_o == 0U) && (d.O_s0 == 1U);

  if (a_contiguous && b_contiguous && o_contiguous) {
    auto *a_store = static_cast<CpuComplexStorage *>(a.storage.get());
    auto *b_store = static_cast<CpuComplexStorage *>(b.storage.get());
    auto *o_store = static_cast<CpuComplexStorage *>(out.storage.get());

    const real1_f alpha[2] = {ONE_R1_F, ZERO_R1_F}; // complex 1+0i
    const real1_f beta[2] = {ZERO_R1_F, ZERO_R1_F}; // complex 0+0i

#if WEED_FPPOW == 5
    cblas_cgemm(
#else
    cblas_zgemm(
#endif
        CblasColMajor, CblasNoTrans, CblasNoTrans, (blasint)d.M, (blasint)d.N,
        (blasint)d.K, alpha, a_store->data.get(), (blasint)d.A_s1,
        b_store->data.get(), (blasint)d.B_s1, beta, o_store->data.get(),
        (blasint)d.O_s1);
    return;
  }
#endif
  cpu<ComplexStorage, ComplexStorage, ComplexStorage, complex>(a, b, out);
}
static inline void cpu_mixed_c_left(const Tensor &a, const Tensor &b,
                                    Tensor &out) {
  cpu<ComplexStorage, RealStorage, ComplexStorage, complex>(a, b, out);
}
static inline void cpu_mixed_c_right(const Tensor &a, const Tensor &b,
                                     Tensor &out) {
  cpu<RealStorage, ComplexStorage, ComplexStorage, complex>(a, b, out);
}

#if ENABLE_GPU
static void gpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
  TILE_BY_TYPE(GpuRealStorage, GpuRealStoragePtr, GpuRealStorage,
               GpuRealStoragePtr, GpuRealStorage, GpuRealStoragePtr,
               OCL_API_MATMUL_REAL);
}
static void gpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
  TILE_BY_TYPE(GpuComplexStorage, GpuComplexStoragePtr, GpuComplexStorage,
               GpuComplexStoragePtr, GpuComplexStorage, GpuComplexStoragePtr,
               OCL_API_MATMUL_COMPLEX);
}
static void gpu_mixed_c_left(const Tensor &a, const Tensor &b, Tensor &out) {
  TILE_BY_TYPE(GpuComplexStorage, GpuComplexStoragePtr, GpuRealStorage,
               GpuRealStoragePtr, GpuComplexStorage, GpuComplexStoragePtr,
               OCL_API_MATMUL_MIXED_C_LEFT);
}
static void gpu_mixed_c_right(const Tensor &a, const Tensor &b, Tensor &out) {
  TILE_BY_TYPE(GpuRealStorage, GpuRealStoragePtr, GpuComplexStorage,
               GpuComplexStoragePtr, GpuComplexStorage, GpuComplexStoragePtr,
               OCL_API_MATMUL_MIXED_C_RIGHT);
}
#endif

void matmul(const Tensor &a, const Tensor &b, Tensor &out) {
  validate_all_same_device({&a, &b, &out}, "MatMulKernel::matmul");
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
} // namespace Weed
