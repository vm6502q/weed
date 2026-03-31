//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of
// Qrack.
//
// This file was produced by (Anthropic) Claude based on softmax.cpp.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#include "ops/logsoftmax.hpp"
#include "common/parallel_for.hpp"
#include "ops/util.hpp"
#include "tensors/flat_tensors.hpp"

// ---------------------------------------------------------------------------
// Index computation macros — identical to softmax.cpp
// ---------------------------------------------------------------------------
#define LOGSOFTMAX_HEAD                                                        \
  tcapint base = a.offset;                                                     \
  tcapint tmp = o;                                                             \
  for (int64_t d = (int64_t)a.shape.size() - 1; d >= 0; --d) {                 \
    if (d == id) {                                                             \
      continue;                                                                \
    }                                                                          \
    const tcapint dim = a.shape[d];                                            \
    const tcapint i_d = tmp % dim;                                             \
    tmp /= dim;                                                                \
    base += i_d * a.stride[d];                                                 \
  }                                                                            \
  tcapint out_base = out.offset;                                               \
  tmp = o;                                                                     \
  for (int64_t d = (int64_t)out.shape.size() - 1; d >= 0; --d) {               \
    if (d == id) {                                                             \
      continue;                                                                \
    }                                                                          \
    const tcapint dim = out.shape[d];                                          \
    const tcapint i_d = tmp % dim;                                             \
    tmp /= dim;                                                                \
    out_base += i_d * out.stride[d];                                           \
  }

#define LOGSOFTMAX_BWD_HEAD                                                    \
  tcapint din_base = din.offset;                                               \
  tcapint tmp = o;                                                             \
  for (int64_t d = (int64_t)din.shape.size() - 1; d >= 0; --d) {               \
    if (d == id) {                                                             \
      continue;                                                                \
    }                                                                          \
    const tcapint dim = din.shape[d];                                          \
    const tcapint i_d = tmp % dim;                                             \
    tmp /= dim;                                                                \
    din_base += i_d * din.stride[d];                                           \
  }                                                                            \
  tcapint out_base = out.offset;                                               \
  tmp = o;                                                                     \
  for (int64_t d = (int64_t)out.shape.size() - 1; d >= 0; --d) {               \
    if (d == id) {                                                             \
      continue;                                                                \
    }                                                                          \
    const tcapint dim = out.shape[d];                                          \
    const tcapint i_d = tmp % dim;                                             \
    tmp /= dim;                                                                \
    out_base += i_d * out.stride[d];                                           \
  }                                                                            \
  tcapint dout_base = dout.offset;                                             \
  tmp = o;                                                                     \
  for (int64_t d = (int64_t)dout.shape.size() - 1; d >= 0; --d) {              \
    if (d == id) {                                                             \
      continue;                                                                \
    }                                                                          \
    const tcapint dim = dout.shape[d];                                         \
    const tcapint i_d = tmp % dim;                                             \
    tmp /= dim;                                                                \
    dout_base += i_d * dout.stride[d];                                         \
  }

// ---------------------------------------------------------------------------
// Forward loop:
//   Pass 1: max for numerical stability
//   Pass 2: sum of exp(x - max)  →  log(sum) = log(s)
//   Pass 3: write (x[j] - max) - log(s)
// ---------------------------------------------------------------------------
#define LOGSOFTMAX_FWD_LOOP(type)                                              \
  const tcapint axis_size = a.shape[id];                                       \
  const tcapint a_stride = a.stride[id];                                       \
  const tcapint out_stride = out.stride[id];                                   \
  /* Pass 1: max */                                                            \
  real1 mx = (*pa)[base];                                                      \
  for (tcapint j = 1U; j < axis_size; ++j) {                                   \
    const real1 v = (*pa)[base + j * a_stride];                                \
    if (v > mx) {                                                              \
      mx = v;                                                                  \
    }                                                                          \
  }                                                                            \
  /* Pass 2: sum of exp(x - max), then log */                                  \
  type s = ZERO_R1;                                                            \
  for (tcapint j = 0U; j < axis_size; ++j) {                                   \
    s += std::exp((*pa)[base + j * a_stride] - mx);                            \
  }                                                                            \
  const real1 log_s = std::log((real1)s);                                      \
  /* Pass 3: write (x - max) - log(sum) */                                     \
  for (tcapint j = 0U; j < axis_size; ++j) {                                   \
    po->write(out_base + j * out_stride,                                       \
              ((*pa)[base + j * a_stride] - mx) - log_s);                      \
  }

// ---------------------------------------------------------------------------
// Backward loop:
//   sum_dout = sum(dout, axis)
//   din += dout - exp(lsm) * sum_dout
// where lsm = out (the stored log-softmax output)
// ---------------------------------------------------------------------------
#define LOGSOFTMAX_BWD_LOOP(type)                                              \
  const tcapint axis_size = din.shape[id];                                     \
  const tcapint din_stride = din.stride[id];                                   \
  const tcapint out_stride = out.stride[id];                                   \
  const tcapint dout_stride = dout.stride[id];                                 \
  /* Pass 1: sum(dout) along axis */                                           \
  type sum_dout = ZERO_R1;                                                     \
  for (tcapint j = 0U; j < axis_size; ++j) {                                   \
    sum_dout += (*pdout)[dout_base + j * dout_stride];                         \
  }                                                                            \
  /* Pass 2: din += dout - exp(lsm) * sum_dout */                              \
  for (tcapint j = 0U; j < axis_size; ++j) {                                   \
    pdin->add(din_base + j * din_stride,                                       \
              (*pdout)[dout_base + j * dout_stride] -                          \
                  std::exp((real1)(*po)[out_base + j * out_stride]) *          \
                      sum_dout);                                               \
  }

#define LOGSOFTMAX_FWD_KERNEL(type)                                            \
  const int64_t id = (int64_t)index;                                           \
  const tcapint n = out.get_broadcast_size() / out.shape[id];                  \
  pfControl.par_for(0, n, [&](const tcapint &o, const unsigned &cpu) {         \
    LOGSOFTMAX_HEAD                                                            \
    LOGSOFTMAX_FWD_LOOP(type)                                                  \
  });

#define LOGSOFTMAX_BWD_KERNEL(type)                                            \
  const int64_t id = (int64_t)index;                                           \
  const tcapint n = din.get_broadcast_size() / din.shape[id];                  \
  pfControl.par_for(0, n, [&](const tcapint &o, const unsigned &cpu) {         \
    LOGSOFTMAX_BWD_HEAD                                                        \
    LOGSOFTMAX_BWD_LOOP(type)                                                  \
  });

#if ENABLE_GPU
#define DISPATCH_GPU_LOGSOFTMAX_FWD(type, api_call)                            \
  const tcapint args[12U]{a.offset,                                            \
                          (tcapint)index,                                      \
                          out.offset,                                          \
                          out.stride[0U],                                      \
                          a.shape[index],                                      \
                          a.stride[index],                                     \
                          out.stride[index],                                   \
                          0U,                                                  \
                          0U,                                                  \
                          0U,                                                  \
                          0U,                                                  \
                          0U};                                                 \
  std::shared_ptr<type> a_storage =                                            \
      std::dynamic_pointer_cast<type>(a.storage);                              \
  std::shared_ptr<type> o_storage =                                            \
      std::dynamic_pointer_cast<type>(out.storage);                            \
  const cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;          \
  BufferPtr shapeBuffer = a_storage->dev->MakeBuffer(                          \
      flags, sizeof(tcapint) * a.shape.size(), (void *)&(a.shape[0U]));        \
  BufferPtr strideBuffer = a_storage->dev->MakeBuffer(                         \
      flags, sizeof(tcapint) * a.stride.size(), (void *)&(a.stride[0U]));      \
  BufferPtr outStrideBuffer = a_storage->dev->MakeBuffer(                      \
      flags, sizeof(tcapint) * out.stride.size(), (void *)&(out.stride[0U]));  \
  a_storage->dev->RequestKernel(api_call, args,                                \
                                out.get_broadcast_size() / out.shape[index],   \
                                {a_storage->buffer, o_storage->buffer,         \
                                 shapeBuffer, strideBuffer, outStrideBuffer})

#define DISPATCH_GPU_LOGSOFTMAX_BWD(din_type, dout_type, api_call)             \
  const tcapint args[12U]{din.offset,                                          \
                          (tcapint)index,                                      \
                          out.offset,                                          \
                          dout.offset,                                         \
                          din.shape[index],                                    \
                          din.stride[index],                                   \
                          out.stride[index],                                   \
                          dout.stride[index],                                  \
                          0U,                                                  \
                          0U,                                                  \
                          0U,                                                  \
                          0U};                                                 \
  std::shared_ptr<din_type> din_storage =                                      \
      std::dynamic_pointer_cast<din_type>(din.storage);                        \
  std::shared_ptr<GpuRealStorage> out_storage =                                \
      std::dynamic_pointer_cast<GpuRealStorage>(out.storage);                  \
  std::shared_ptr<dout_type> dout_storage =                                    \
      std::dynamic_pointer_cast<dout_type>(dout.storage);                      \
  const cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;          \
  BufferPtr shapeBuffer = din_storage->dev->MakeBuffer(                        \
      flags, sizeof(tcapint) * din.shape.size(), (void *)&(din.shape[0U]));    \
  BufferPtr strideBuffer = din_storage->dev->MakeBuffer(                       \
      flags, sizeof(tcapint) * din.stride.size(), (void *)&(din.stride[0U]));  \
  din_storage->dev->RequestKernel(                                             \
      api_call, args, din.get_broadcast_size() / din.shape[index],             \
      {din_storage->buffer, out_storage->buffer, dout_storage->buffer,         \
       shapeBuffer, strideBuffer})
#endif

#define DEVICE_SWITCH_LOGSOFTMAX_FWD(cpu, gpu)                                 \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(index, a, out);                                                        \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(index, a, out);                                                        \
  }

#define DEVICE_SWITCH_LOGSOFTMAX_BWD(cpu, gpu)                                 \
  switch (din.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(index, din, out, dout);                                                \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(index, din, out, dout);                                                \
  }

namespace Weed {

// ---------------------------------------------------------------------------
// CPU forward
// ---------------------------------------------------------------------------
static void cpu_logsoftmax_fwd(const tcapint &index, const Tensor &a,
                               Tensor &out) {
  GET_CONST_FLAT_TENSOR(RealTensor, a, pa);
  GET_FLAT_TENSOR(RealTensor, out, po);
  LOGSOFTMAX_FWD_KERNEL(real1)
}

// ---------------------------------------------------------------------------
// CPU backward
// ---------------------------------------------------------------------------
template <typename T1, typename T2, typename T3>
static void cpu_logsoftmax_bwd(const tcapint &index, Tensor &din,
                               const Tensor &out, const Tensor &dout) {
  GET_FLAT_TENSOR(T1, din, pdin);
  GET_CONST_FLAT_TENSOR(RealTensor, out, po);
  GET_CONST_FLAT_TENSOR(T2, dout, pdout);
  LOGSOFTMAX_BWD_KERNEL(T3)
}
static void cpu_logsoftmax_bwd_real(const tcapint &index, Tensor &din,
                                    const Tensor &out, const Tensor &dout) {
  cpu_logsoftmax_bwd<RealTensor, RealTensor, real1>(index, din, out, dout);
}
static void cpu_logsoftmax_bwd_complex(const tcapint &index, Tensor &din,
                                       const Tensor &out, const Tensor &dout) {
  cpu_logsoftmax_bwd<ComplexTensor, ComplexTensor, complex>(index, din, out,
                                                            dout);
}
static void cpu_logsoftmax_bwd_mixed(const tcapint &index, Tensor &din,
                                     const Tensor &out, const Tensor &dout) {
  cpu_logsoftmax_bwd<ComplexTensor, RealTensor, complex>(index, din, out, dout);
}

// ---------------------------------------------------------------------------
// GPU stubs (kernels to be added to qenginecl.cl)
// ---------------------------------------------------------------------------
#if ENABLE_GPU
static void gpu_logsoftmax_fwd(const tcapint &index, const Tensor &a,
                               Tensor &out) {
  DISPATCH_GPU_LOGSOFTMAX_FWD(GpuRealStorage, OCL_API_LOGSOFTMAX);
}

static void gpu_logsoftmax_bwd_real(const tcapint &index, Tensor &din,
                                    const Tensor &out, const Tensor &dout) {
  DISPATCH_GPU_LOGSOFTMAX_BWD(GpuRealStorage, GpuRealStorage,
                              OCL_API_LOGSOFTMAX_BACKWARD_REAL);
}
static void gpu_logsoftmax_bwd_complex(const tcapint &index, Tensor &din,
                                       const Tensor &out, const Tensor &dout) {
  DISPATCH_GPU_LOGSOFTMAX_BWD(GpuComplexStorage, GpuComplexStorage,
                              OCL_API_LOGSOFTMAX_BACKWARD_COMPLEX);
}
static void gpu_logsoftmax_bwd_mixed(const tcapint &index, Tensor &din,
                                     const Tensor &out, const Tensor &dout) {
  DISPATCH_GPU_LOGSOFTMAX_BWD(GpuComplexStorage, GpuRealStorage,
                              OCL_API_LOGSOFTMAX_BACKWARD_MIXED);
}
#endif

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
void logsoftmax(const tcapint &index, const Tensor &a, Tensor &out) {
  validate_all_same_device({&a, &out}, "LogSoftmaxKernel::logsoftmax");
  if ((a.storage->dtype != DType::REAL) ||
      (out.storage->dtype != DType::REAL)) {
    throw std::invalid_argument("Tensor dtype mismatch in logsoftmax!");
  }
#if ENABLE_GPU
  DEVICE_SWITCH_LOGSOFTMAX_FWD(cpu_logsoftmax_fwd, gpu_logsoftmax_fwd);
#else
  cpu_logsoftmax_fwd(index, a, out);
#endif
}

void logsoftmax_grad(const tcapint &index, Tensor &din, const Tensor &out,
                     const Tensor &dout) {
  validate_all_same_device({&din, &out, &dout},
                           "LogSoftmaxKernel::logsoftmax_grad");
  if (out.storage->dtype != DType::REAL) {
    throw std::invalid_argument(
        "Tensor dtype mismatch in logsoftmax_grad: out must be real!");
  }
  if ((din.storage->dtype == DType::REAL) &&
      (dout.storage->dtype != DType::REAL)) {
    throw std::invalid_argument(
        "In Weed::logsoftmax_grad(din, out, dout), dout dtype "
        "must upcast to din dtype!");
  }
  switch (din.storage->dtype) {
  case DType::COMPLEX:
    switch (dout.storage->dtype) {
    case DType::COMPLEX:
#if ENABLE_GPU
      DEVICE_SWITCH_LOGSOFTMAX_BWD(cpu_logsoftmax_bwd_complex,
                                   gpu_logsoftmax_bwd_complex);
#else
      cpu_logsoftmax_bwd_complex(index, din, out, dout);
#endif
      break;
    case DType::REAL:
    default:
#if ENABLE_GPU
      DEVICE_SWITCH_LOGSOFTMAX_BWD(cpu_logsoftmax_bwd_mixed,
                                   gpu_logsoftmax_bwd_mixed);
#else
      cpu_logsoftmax_bwd_mixed(index, din, out, dout);
#endif
    }
    break;
  case DType::REAL:
  default:
#if ENABLE_GPU
    DEVICE_SWITCH_LOGSOFTMAX_BWD(cpu_logsoftmax_bwd_real,
                                 gpu_logsoftmax_bwd_real);
#else
    cpu_logsoftmax_bwd_real(index, din, out, dout);
#endif
  }
}

} // namespace Weed
