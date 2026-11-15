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

#include "ops/embedding.hpp"
#include "common/parallel_for.hpp"
#include "ops/util.hpp"
#include "storage/all_storage.hpp"
#if ENABLE_GPU
#include "ops/gpu_flat.hpp"
#include "storage/gpu_int_storage.hpp"
#endif

#include <iostream>

#define DEVICE_SWITCH(cpu, gpu, a, b, o)                                       \
  switch (a.storage->device) {                                                 \
  case DeviceTag::GPU:                                                         \
    gpu(a, b, o);                                                              \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, b, o);                                                              \
  }

// Kernel arguments (see qengine.cl): the indices are (offset, step); "table" is
// the (vocabulary, D) weight or its gradient, with (offset, row stride, column
// stride); and "act" is the contiguous (indices..., D) output or its gradient,
// with its offset and its column stride, which is the number of indices, N.
#define GPU_EMBEDDING_ARGS(table, act)                                         \
  if (!gpu_is_flat(indices)) {                                                 \
    throw std::invalid_argument(                                               \
        "GPU embedding indices must be contiguous (or a single index)!");      \
  }                                                                            \
  if (table.shape.size() != 2U) {                                              \
    throw std::invalid_argument("Embedding weight must have 2 indices!");      \
  }                                                                            \
  const tcapint N = indices.get_broadcast_size();                              \
  const tcapint D = table.shape[1U];                                           \
  const tcapint args[12U] {                                                    \
    indices.offset, gpu_step(indices), table.offset, 0U, table.stride[1U], N,  \
        act.offset, D, table.stride[0U], N, 0U, 0U                             \
  }

#define DISPATCH_GPU_FORWARD(type, api_call)                                   \
  if (!gpu_is_flat(out) || (out.is_scalar() && (out.get_size() > 1U))) {       \
    throw std::invalid_argument("GPU embedding output must be contiguous!");   \
  }                                                                            \
  GPU_EMBEDDING_ARGS(weight, out);                                             \
  std::shared_ptr<GpuIntStorage> i_storage =                                   \
      std::dynamic_pointer_cast<GpuIntStorage>(indices.storage);               \
  std::shared_ptr<type> w_storage =                                            \
      std::dynamic_pointer_cast<type>(weight.storage);                         \
  std::shared_ptr<type> o_storage =                                            \
      std::dynamic_pointer_cast<type>(out.storage);                            \
  i_storage->dev->RequestKernel(                                               \
      api_call, args, N,                                                       \
      {i_storage->buffer, w_storage->buffer, o_storage->buffer})

// One work item per column of the table: see the kernel.
#define DISPATCH_GPU_BACKWARD(type1, type2, api_call)                          \
  const Tensor fdout = gpu_contiguous(dout);                                   \
  GPU_EMBEDDING_ARGS(dW, fdout);                                               \
  std::shared_ptr<GpuIntStorage> i_storage =                                   \
      std::dynamic_pointer_cast<GpuIntStorage>(indices.storage);               \
  std::shared_ptr<type1> w_storage =                                           \
      std::dynamic_pointer_cast<type1>(dW.storage);                            \
  std::shared_ptr<type2> o_storage =                                           \
      std::dynamic_pointer_cast<type2>(fdout.storage);                         \
  i_storage->dev->RequestKernel(                                               \
      api_call, args, D,                                                       \
      {i_storage->buffer, w_storage->buffer, o_storage->buffer})

namespace Weed {
// Storage step between consecutive elements over the first "n" indices of a
// contiguous tensor. (An index of extent 1 has stride 0, so stride[0] alone
// is the wrong step for a shape like (1, T).)
static inline tcapint lead_step(const BaseTensor &t, const size_t &n) {
  for (size_t i = 0U; i < n; ++i) {
    if (t.stride[i]) {
      return t.stride[i];
    }
  }

  return 0U;
}

template <typename T>
static void cpu_forward(const SymbolTensor &indices, const Tensor &weight,
                        Tensor &out) {
  const tcapint D = weight.shape[1];
  const size_t N = indices.get_broadcast_size();

  auto *idx = static_cast<const IntStorage *>(indices.storage.get());
  auto *W = static_cast<const T *>(weight.storage.get());
  auto *O = static_cast<T *>(out.storage.get());

  const tcapint I_s = lead_step(indices, indices.stride.size());
  const tcapint W_s0 = weight.stride[0];
  const tcapint W_s1 = weight.stride[1];
  const tcapint O_s0 = lead_step(out, out.stride.size() - 1U);
  const tcapint O_s1 = out.stride.back();

  pfControl.par_for(0U, N, [&](const tcapint &i, const unsigned &) {
    const tcapint token = (*idx)[indices.offset + i * I_s];

    const tcapint w_base = weight.offset + token * W_s0;
    const tcapint o_base = out.offset + i * O_s0;

    for (tcapint d = 0U; d < D; ++d) {
      O->write(o_base + d * O_s1, (*W)[w_base + d * W_s1]);
    }
  });
}

template <typename T1, typename T2>
static void cpu_backward(Tensor &dW, const SymbolTensor &indices,
                         const Tensor &dout) {
  const tcapint D = dW.shape[1];
  const size_t N = indices.get_broadcast_size();

  auto *idx = static_cast<const IntStorage *>(indices.storage.get());
  auto *dWt = static_cast<T1 *>(dW.storage.get());
  auto *dOut = static_cast<const T2 *>(dout.storage.get());

  const tcapint I_s = lead_step(indices, indices.stride.size());
  const tcapint W_s0 = dW.stride[0];
  const tcapint W_s1 = dW.stride[1];
  const tcapint O_s0 = lead_step(dout, dout.stride.size() - 1U);
  const tcapint O_s1 = dout.stride.back();

  pfControl.par_for(0U, N, [&](const tcapint &i, const unsigned &) {
    const tcapint token = (*idx)[indices.offset + i * I_s];

    const tcapint w_base = dW.offset + token * W_s0;
    const tcapint o_base = dout.offset + i * O_s0;

    for (tcapint d = 0U; d < D; ++d) {
      dWt->add(w_base + d * W_s1, (*dOut)[o_base + d * O_s1]);
    }
  });
}

#if ENABLE_GPU
static void gpu_forward_real(const SymbolTensor &indices, const Tensor &weight,
                             Tensor &out) {
  DISPATCH_GPU_FORWARD(GpuRealStorage, OCL_API_EMBEDDING_REAL);
}
static void gpu_forward_complex(const SymbolTensor &indices,
                                const Tensor &weight, Tensor &out) {
  DISPATCH_GPU_FORWARD(GpuComplexStorage, OCL_API_EMBEDDING_COMPLEX);
}
static void gpu_backward_real(Tensor &dW, const SymbolTensor &indices,
                              const Tensor &dout) {
  DISPATCH_GPU_BACKWARD(GpuRealStorage, GpuRealStorage,
                        OCL_API_EMBEDDING_GRAD_REAL);
}
static void gpu_backward_complex(Tensor &dW, const SymbolTensor &indices,
                                 const Tensor &dout) {
  DISPATCH_GPU_BACKWARD(GpuComplexStorage, GpuComplexStorage,
                        OCL_API_EMBEDDING_GRAD_COMPLEX);
}
static void gpu_backward_mixed(Tensor &dW, const SymbolTensor &indices,
                               const Tensor &dout) {
  DISPATCH_GPU_BACKWARD(GpuComplexStorage, GpuRealStorage,
                        OCL_API_EMBEDDING_GRAD_MIXED);
}
#endif

void embedding_gather(const SymbolTensor &a, const Tensor &b, Tensor &o) {
  validate_all_same_device({&a, &b, &o}, "embedding_gather");
  const bool isAComplex = a.storage->dtype == DType::COMPLEX;
  if (isAComplex) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_forward<ComplexStorage>, gpu_forward_complex, a, b, o);
#else
    cpu_forward<ComplexStorage>(a, b, o);
#endif
  } else {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_forward<RealStorage>, gpu_forward_real, a, b, o);
#else
    cpu_forward<RealStorage>(a, b, o);
#endif
  }
}

void embedding_scatter_add(Tensor &dW, const SymbolTensor &indices,
                           const Tensor &dout) {
  validate_all_same_device({&dW, &indices, &dout}, "embedding_scatter_add");
  const bool isAComplex = dW.storage->dtype == DType::COMPLEX;
  const bool isOComplex = dout.storage->dtype == DType::COMPLEX;
  if (isAComplex && !isOComplex) {
    throw std::invalid_argument(
        "Cannot combine complex tensors into real1 tensor!");
  }
  if (isOComplex) {
    if (isAComplex) {
#if ENABLE_GPU
      if (dout.storage->device == DeviceTag::GPU) {
        gpu_backward_complex(dW, indices, dout);
      } else {
        cpu_backward<ComplexStorage, ComplexStorage>(dW, indices, dout);
      }
#else
      cpu_backward<ComplexStorage, ComplexStorage>(dW, indices, dout);
#endif
    } else {
#if ENABLE_GPU
      if (dout.storage->device == DeviceTag::GPU) {
        gpu_backward_mixed(dW, indices, dout);
      } else {
        cpu_backward<ComplexStorage, RealStorage>(dW, indices, dout);
      }
#else
      cpu_backward<ComplexStorage, RealStorage>(dW, indices, dout);
#endif
    }
  } else {
#if ENABLE_GPU
    if (dout.storage->device == DeviceTag::GPU) {
      gpu_backward_real(dW, indices, dout);
    } else {
      cpu_backward<RealStorage, RealStorage>(dW, indices, dout);
    }
#else
    cpu_backward<RealStorage, RealStorage>(dW, indices, dout);
#endif
  }
}
} // namespace Weed
