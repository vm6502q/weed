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

#include "ops/reduce.hpp"
#include "common/parallel_for.hpp"
#include "ops/util.hpp"
#include "tensors/flat_tensors.hpp"

#define REDUCE_HEAD(type)                                                      \
  tcapint base = 0U;                                                           \
  tcapint tmp = o;                                                             \
                                                                               \
  for (int64_t d = a.shape.size() - 1; d >= 0; --d) {                          \
    if (d == id) {                                                             \
      continue;                                                                \
    }                                                                          \
                                                                               \
    tcapint dim = a.shape[d];                                                  \
    tcapint i_d = tmp % dim;                                                   \
    tmp /= dim;                                                                \
                                                                               \
    base += i_d * a.stride[d];                                                 \
  }

#define SUM_LOOP(type)                                                         \
  type sum = ZERO_R1;                                                          \
  for (tcapint j = 0U; j < a.shape[id]; ++j) {                                 \
    sum += (*pa)[base + j * a.stride[id]];                                     \
  }                                                                            \
  po->write(o, sum)

#define MAX_LOOP(type)                                                         \
  type m = (*pa)[base];                                                        \
  for (tcapint j = 1U; j < a.shape[id]; ++j) {                                 \
    const type v = (*pa)[base + j * a.stride[id]];                             \
    if (v > m) {                                                               \
      m = v;                                                                   \
    }                                                                          \
  }                                                                            \
  po->write(o, m)

#define MIN_LOOP(type)                                                         \
  type m = (*pa)[base];                                                        \
  for (tcapint j = 1U; j < a.shape[id]; ++j) {                                 \
    const type v = (*pa)[base + j * a.stride[id]];                             \
    if (v < m) {                                                               \
      m = v;                                                                   \
    }                                                                          \
  }                                                                            \
  po->write(o, m)

#define SUM_KERNEL(type)                                                       \
  const tcapint n = out.get_broadcast_size();                                  \
  const int64_t id = index;                                                    \
  pfControl.par_for(0, n, [&](const tcapint &o, const unsigned &cpu) {         \
    REDUCE_HEAD(type);                                                         \
    SUM_LOOP(type);                                                            \
  });

#define MAX_KERNEL(type)                                                       \
  const tcapint n = out.get_broadcast_size();                                  \
  const int64_t id = index;                                                    \
  pfControl.par_for(0, n, [&](const tcapint &o, const unsigned &cpu) {         \
    REDUCE_HEAD(type);                                                         \
    MAX_LOOP(type);                                                            \
  });

#define MIN_KERNEL(type)                                                       \
  const tcapint n = out.get_broadcast_size();                                  \
  const int64_t id = index;                                                    \
  pfControl.par_for(0, n, [&](const tcapint &o, const unsigned &cpu) {         \
    REDUCE_HEAD(type);                                                         \
    MIN_LOOP(type);                                                            \
  });

#define REDUCE_GRAD_HEAD(type)                                                 \
  tcapint tmp = i;                                                             \
  tcapint o = 0U;                                                              \
                                                                               \
  for (int64_t d = in.shape.size() - 1; d >= 0; --d) {                         \
    if (d == id) {                                                             \
      continue;                                                                \
    }                                                                          \
                                                                               \
    tcapint dim = in.shape[d];                                                 \
    tcapint i_d = tmp % dim;                                                   \
    tmp /= dim;                                                                \
                                                                               \
    o += i_d * dout.stride[d];                                                 \
  }

#define SUM_GRAD_OUT(type) pdi->add(i, (type)(*po)[o])

#define MATCH_GRAD_OUT(type)                                                   \
  if ((*ti)[i] == (*to)[o]) {                                                  \
    pdi->add(i, (type)(*po)[o]);                                               \
  }

#define SUM_GRAD_KERNEL(type)                                                  \
  const tcapint n = din.get_broadcast_size();                                  \
  const int64_t id = index;                                                    \
  pfControl.par_for(0, n, [&](const tcapint &i, const unsigned &cpu) {         \
    REDUCE_GRAD_HEAD(type);                                                    \
    SUM_GRAD_OUT(type);                                                        \
  });

#define MATCH_GRAD_KERNEL(type)                                                \
  const tcapint n = din.get_broadcast_size();                                  \
  const int64_t id = index;                                                    \
  pfControl.par_for(0, n, [&](const tcapint &i, const unsigned &cpu) {         \
    REDUCE_GRAD_HEAD(type);                                                    \
    MATCH_GRAD_OUT(type);                                                      \
  });

#define DISPATCH_GPU_KERNEL(type, api_call)                                    \
  const tcapint args[10U]{a.offset,   (tcapint)index,                          \
                          out.offset, out.stride[0U],                          \
                          0U,         0U,                                      \
                          0U,         0U,                                      \
                          0U,         0U};                                     \
  std::shared_ptr<type> a_storage =                                            \
      std::dynamic_pointer_cast<type>(a.storage);                              \
  std::shared_ptr<type> o_storage =                                            \
      std::dynamic_pointer_cast<type>(out.storage);                            \
  const cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;          \
  BufferPtr shapeBuffer = a_storage->dev->MakeBuffer(                          \
      flags, sizeof(tcapint) * a.shape.size(), (void *)&(a.shape[0U]));        \
  BufferPtr strideBuffer = a_storage->dev->MakeBuffer(                         \
      flags, sizeof(tcapint) * a.stride.size(), (void *)&(a.stride[0U]));      \
  a_storage->dev->RequestKernel(                                               \
      api_call, args, out.get_size(),                                          \
      {a_storage->buffer, o_storage->buffer, shapeBuffer, strideBuffer})

#define DEVICE_SWITCH(cpu, gpu)                                                \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(index, a, out);                                                        \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(index, a, out);                                                        \
  }

#define DEVICE_SWITCH_SUM_GRAD(cpu, gpu)                                       \
  switch (din.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(index, din, in, dout);                                                 \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(index, din, in, dout);                                                 \
  }

#define DEVICE_SWITCH_MATCH_GRAD(cpu, gpu)                                     \
  switch (din.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(index, din, in, dout, out);                                            \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(index, din, in, dout, out);                                            \
  }

#define CPU_SUM_GRAD(t1, t2, type)                                             \
  GET_FLAT_TENSOR(t1, din, pdi);                                               \
  GET_CONST_FLAT_TENSOR(t2, dout, po);                                         \
  SUM_GRAD_KERNEL(type)

#define CPU_MATCH_GRAD(t1, t2, type)                                           \
  GET_FLAT_TENSOR(t1, din, pdi);                                               \
  GET_CONST_FLAT_TENSOR(RealTensor, in, ti);                                   \
  GET_CONST_FLAT_TENSOR(t2, dout, po);                                         \
  GET_CONST_FLAT_TENSOR(RealTensor, out, to);                                  \
  MATCH_GRAD_KERNEL(type)

#define GPU_SUM_GRAD_ARGS()                                                    \
  const tcapint args[10U] {                                                    \
    din.offset, din.stride[0U], in.offset, in.stride[0U], dout.offset,         \
        dout.stride[0U], 0U, 0U, 0U, 0U                                        \
  }

#define GPU_MATCH_GRAD_ARGS()                                                  \
  const tcapint args[10U] {                                                    \
    din.offset, din.stride[0U], in.offset, in.stride[0U], dout.offset,         \
        dout.stride[0U], out.offset, out.stride[0U], 0U, 0U                    \
  }

#define GPU_SUM_GRAD(type1, type2, api_call)                                   \
  GPU_SUM_GRAD_ARGS();                                                         \
  std::shared_ptr<type1> a_storage =                                           \
      std::dynamic_pointer_cast<type1>(din.storage);                           \
  std::shared_ptr<type2> b_storage =                                           \
      std::dynamic_pointer_cast<type2>(dout.storage);                          \
  const cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;          \
  BufferPtr shapeBuffer = a_storage->dev->MakeBuffer(                          \
      flags, sizeof(tcapint) * in.shape.size(), (void *)&(in.shape[0U]));      \
  BufferPtr strideBuffer = a_storage->dev->MakeBuffer(                         \
      flags, sizeof(tcapint) * in.stride.size(), (void *)&(in.stride[0U]));    \
  a_storage->dev->RequestKernel(                                               \
      OCLAPI::api_call, args, din.get_size(),                                  \
      {a_storage->buffer, b_storage->buffer, shapeBuffer, strideBuffer})

#define GPU_MATCH_GRAD(type1, type2, api_call)                                 \
  GPU_SUM_GRAD_ARGS();                                                         \
  std::shared_ptr<type1> a_storage =                                           \
      std::dynamic_pointer_cast<type1>(din.storage);                           \
  std::shared_ptr<type2> b_storage =                                           \
      std::dynamic_pointer_cast<type2>(dout.storage);                          \
  GpuRealStoragePtr i_storage =                                                \
      std::dynamic_pointer_cast<GpuRealStorage>(in.storage);                   \
  GpuRealStoragePtr o_storage =                                                \
      std::dynamic_pointer_cast<GpuRealStorage>(out.storage);                  \
  const cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;          \
  BufferPtr shapeBuffer = a_storage->dev->MakeBuffer(                          \
      flags, sizeof(tcapint) * in.shape.size(), (void *)&(in.shape[0U]));      \
  BufferPtr strideBuffer = a_storage->dev->MakeBuffer(                         \
      flags, sizeof(tcapint) * in.stride.size(), (void *)&(in.stride[0U]));    \
  a_storage->dev->RequestKernel(OCLAPI::api_call, args, din.get_size(),        \
                                {a_storage->buffer, b_storage->buffer,         \
                                 i_storage->buffer, o_storage->buffer,         \
                                 shapeBuffer, strideBuffer})

namespace Weed {
template <typename T1, typename T2, typename T3>
static void cpu_sum(const tcapint &index, const Tensor &a, Tensor &out) {
  GET_CONST_FLAT_TENSOR(T1, a, pa);
  GET_STORAGE(T2, out, po);
  SUM_KERNEL(T3);
}
static void cpu_max(const tcapint &index, const Tensor &a, Tensor &out) {
  GET_CONST_FLAT_TENSOR(RealTensor, a, pa);
  GET_STORAGE(RealStorage, out, po);
  MAX_KERNEL(real1);
}
static void cpu_min(const tcapint &index, const Tensor &a, Tensor &out) {
  GET_CONST_FLAT_TENSOR(RealTensor, a, pa);
  GET_STORAGE(RealStorage, out, po);
  MIN_KERNEL(real1);
}
static inline void cpu_sum_real(const tcapint &index, const Tensor &a,
                                Tensor &out) {
  cpu_sum<RealTensor, RealStorage, real1>(index, a, out);
}
static inline void cpu_sum_complex(const tcapint &index, const Tensor &a,
                                   Tensor &out) {
  cpu_sum<ComplexTensor, ComplexStorage, complex>(index, a, out);
}
template <typename T1, typename T2, typename T3>
static void cpu_sum_grad(const tcapint &index, Tensor &din, const Tensor &in,
                         const Tensor &dout) {
  CPU_SUM_GRAD(T1, T2, T3);
}
template <typename T1, typename T2, typename T3>
static void cpu_match_grad(const tcapint &index, Tensor &din, const Tensor &in,
                           const Tensor &dout, const Tensor &out) {
  CPU_MATCH_GRAD(T1, T2, T3);
}
static inline void cpu_sum_grad_real(const tcapint &index, Tensor &din,
                                     const Tensor &in, const Tensor &dout) {
  cpu_sum_grad<RealTensor, RealTensor, real1>(index, din, in, dout);
}
static inline void cpu_sum_grad_complex(const tcapint &index, Tensor &din,
                                        const Tensor &in, const Tensor &dout) {
  cpu_sum_grad<ComplexTensor, ComplexTensor, complex>(index, din, in, dout);
}
static inline void cpu_sum_grad_mixed(const tcapint &index, Tensor &din,
                                      const Tensor &in, const Tensor &dout) {
  cpu_sum_grad<ComplexTensor, RealTensor, complex>(index, din, in, dout);
}
static inline void cpu_match_grad_real(const tcapint &index, Tensor &din,
                                       const Tensor &in, const Tensor &dout,
                                       const Tensor &out) {
  cpu_match_grad<RealTensor, RealTensor, real1>(index, din, in, dout, out);
}
static inline void cpu_match_grad_complex(const tcapint &index, Tensor &din,
                                          const Tensor &in, const Tensor &dout,
                                          const Tensor &out) {
  cpu_match_grad<ComplexTensor, ComplexTensor, complex>(index, din, in, dout,
                                                        out);
}
static inline void cpu_match_grad_mixed(const tcapint &index, Tensor &din,
                                        const Tensor &in, const Tensor &dout,
                                        const Tensor &out) {
  cpu_match_grad<ComplexTensor, RealTensor, complex>(index, din, in, dout, out);
}

#if ENABLE_GPU
static void gpu_sum_real(const tcapint &index, const Tensor &a, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, OCL_API_REDUCE_REAL);
}
static void gpu_sum_complex(const tcapint &index, const Tensor &a,
                            Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, OCL_API_REDUCE_COMPLEX);
}
static void gpu_max(const tcapint &index, const Tensor &a, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, OCL_API_AXIS_MAX);
}
static void gpu_min(const tcapint &index, const Tensor &a, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, OCL_API_AXIS_MIN);
}
static void gpu_sum_grad_real(const tcapint &index, Tensor &din,
                              const Tensor &in, const Tensor &dout) {
  GPU_SUM_GRAD(GpuRealStorage, GpuRealStorage, OCL_API_REDUCE_GRAD_REAL);
}
static void gpu_sum_grad_complex(const tcapint &index, Tensor &din,
                                 const Tensor &in, const Tensor &dout) {
  GPU_SUM_GRAD(GpuComplexStorage, GpuComplexStorage,
               OCL_API_REDUCE_GRAD_COMPLEX);
}
static void gpu_sum_grad_mixed(const tcapint &index, Tensor &din,
                               const Tensor &in, const Tensor &dout) {
  GPU_SUM_GRAD(GpuRealStorage, GpuComplexStorage, OCL_API_REDUCE_GRAD_MIXED);
}
static void gpu_match_grad_real(const tcapint &index, Tensor &din,
                                const Tensor &in, const Tensor &dout,
                                const Tensor &out) {
  GPU_MATCH_GRAD(GpuRealStorage, GpuRealStorage, OCL_API_AXIS_MATCH_GRAD_REAL);
}
static void gpu_match_grad_complex(const tcapint &index, Tensor &din,
                                   const Tensor &in, const Tensor &dout,
                                   const Tensor &out) {
  GPU_MATCH_GRAD(GpuComplexStorage, GpuComplexStorage,
                 OCL_API_AXIS_MATCH_GRAD_COMPLEX);
}
static void gpu_match_grad_mixed(const tcapint &index, Tensor &din,
                                 const Tensor &in, const Tensor &dout,
                                 const Tensor &out) {
  GPU_MATCH_GRAD(GpuRealStorage, GpuComplexStorage,
                 OCL_API_AXIS_MATCH_GRAD_MIXED);
}
#endif

void reduce(const tcapint &index, const Tensor &a, Tensor &out) {
  validate_all_same_device({&a, &out}, "ReduceKernel::reduce");
  if (a.storage->dtype != out.storage->dtype) {
    throw std::invalid_argument("Output tensor dtype mismatch in reduce!");
  }
  if (a.storage->dtype == DType::COMPLEX) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_sum_complex, gpu_sum_complex);
#else
    cpu_sum_complex(index, a, out);
#endif
  } else {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_sum_real, gpu_sum_real);
#else
    cpu_sum_real(index, a, out);
#endif
  }
}

void max(const tcapint &index, const Tensor &a, Tensor &out) {
  validate_all_same_device({&a, &out}, "ReduceKernel::max");
  if ((a.storage->dtype != DType::REAL) ||
      (out.storage->dtype != DType::REAL)) {
    throw std::invalid_argument("Tensor dtype mismatch in max!");
  }
#if ENABLE_GPU
  DEVICE_SWITCH(cpu_max, gpu_max);
#else
  cpu_max(index, a, out);
#endif
}

void min(const tcapint &index, const Tensor &a, Tensor &out) {
  validate_all_same_device({&a, &out}, "ReduceKernel::min");
  if ((a.storage->dtype != DType::REAL) ||
      (out.storage->dtype != DType::REAL)) {
    throw std::invalid_argument("Tensor dtype mismatch in min!");
  }
#if ENABLE_GPU
  DEVICE_SWITCH(cpu_min, gpu_min);
#else
  cpu_min(index, a, out);
#endif
}

void reduce_grad(const tcapint &index, Tensor &din, const Tensor &in,
                 const Tensor &dout) {
  validate_all_same_device({&din, &dout}, "ReduceKernel::reduce_grad");
  if ((din.storage->dtype == DType::REAL) &&
      (dout.storage->dtype != DType::REAL)) {
    throw std::invalid_argument(
        "In Weed::reduce_grad(din, in, dout), dout dtype "
        "must upcast to dout dtype!");
  }
  const tcapint dinSize = din.get_broadcast_size();
  const tcapint inSize = in.get_broadcast_size();
  const tcapint doutSize = dout.get_broadcast_size();
  if ((dinSize != inSize) || (dinSize != doutSize)) {
    throw std::invalid_argument(
        "In Weed::reduce_grad(din, in, dout), sizes do not match!");
  }
  switch (din.storage->dtype) {
  case DType::COMPLEX:
    switch (dout.storage->dtype) {
    case DType::COMPLEX:
#if ENABLE_GPU
      DEVICE_SWITCH_SUM_GRAD(cpu_sum_grad_complex, gpu_sum_grad_complex);
#else
      cpu_sum_grad_complex(index, din, in, dout);
#endif
      break;
    case DType::REAL:
    default:
#if ENABLE_GPU
      DEVICE_SWITCH_SUM_GRAD(cpu_sum_grad_mixed, gpu_sum_grad_mixed);
#else
      cpu_sum_grad_mixed(index, din, in, dout);
#endif
    }
    break;
  case DType::REAL:
  default:
#if ENABLE_GPU
    DEVICE_SWITCH_SUM_GRAD(cpu_sum_grad_real, gpu_sum_grad_real);
#else
    cpu_sum_grad_real(index, din, in, dout);
#endif
  }
}

void match_grad(const tcapint &index, Tensor &din, const Tensor &in,
                const Tensor &dout, const Tensor &out) {
  validate_all_same_device({&din, &dout}, "ReduceKernel::match_grad");
  if ((in.storage->dtype != DType::REAL) ||
      (out.storage->dtype != DType::REAL)) {
    throw std::invalid_argument("Tensor dtype mismatch in match_grad!");
  }
  if ((din.storage->dtype == DType::REAL) &&
      (dout.storage->dtype != DType::REAL)) {
    throw std::invalid_argument(
        "In Weed::reduce_grad(din, in, dout), dout dtype "
        "must upcast to dout dtype!");
  }
  const tcapint dinSize = din.get_broadcast_size();
  const tcapint inSize = in.get_broadcast_size();
  const tcapint doutSize = dout.get_broadcast_size();
  if ((dinSize != inSize) || (dinSize != doutSize)) {
    throw std::invalid_argument(
        "In Weed::reduce_grad(din, in, dout), sizes do not match!");
  }
  switch (din.storage->dtype) {
  case DType::COMPLEX:
    switch (dout.storage->dtype) {
    case DType::COMPLEX:
#if ENABLE_GPU
      DEVICE_SWITCH_MATCH_GRAD(cpu_match_grad_complex, gpu_match_grad_complex);
#else
      cpu_match_grad_complex(index, din, in, dout);
#endif
      break;
    case DType::REAL:
    default:
#if ENABLE_GPU
      DEVICE_SWITCH_MATCH_GRAD(cpu_match_grad_mixed, gpu_match_grad_mixed);
#else
      cpu_match_grad_mixed(index, din, in, dout);
#endif
    }
    break;
  case DType::REAL:
  default:
#if ENABLE_GPU
    DEVICE_SWITCH_MATCH_GRAD(cpu_match_grad_real, gpu_match_grad_real);
#else
    cpu_match_grad_real(index, din, in, dout);
#endif
  }
}
} // namespace Weed
