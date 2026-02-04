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
#include "storage/gpu_int_storage.hpp"

#define DEVICE_SWITCH(cpu, gpu, a, b, o)                                       \
  switch (a.storage->device) {                                                 \
  case DeviceTag::GPU:                                                         \
    gpu(a, b, o);                                                              \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, b, o);                                                              \
  }

#define DISPATCH_GPU_KERNEL(type1, type2, type3, api_call)                     \
  const tcapint args[10U]{indices.offset,                                      \
                          indices.stride[0U],                                  \
                          weight.offset,                                       \
                          weight.stride[0U],                                   \
                          out.stride.back(),                                   \
                          weight.stride[1U],                                   \
                          out.offset,                                          \
                          weight.shape[1U],                                    \
                          0U,                                                  \
                          0U};                                                 \
  std::shared_ptr<type1> a_storage =                                           \
      std::dynamic_pointer_cast<type1>(indices.storage);                       \
  std::shared_ptr<type2> b_storage =                                           \
      std::dynamic_pointer_cast<type2>(weight.storage);                        \
  std::shared_ptr<type3> c_storage =                                           \
      std::dynamic_pointer_cast<type3>(out.storage);                           \
  a_storage->dev->RequestKernel(                                               \
      api_call, args, indices.get_broadcast_size(),                            \
      {a_storage->buffer, b_storage->buffer, c_storage->buffer})

namespace Weed {
template <typename T>
void cpu_forward(const SymbolTensor &indices, const Tensor &weight,
                 Tensor &out) {
  const tcapint D = weight.shape[1];
  const size_t N = indices.get_broadcast_size();

  auto *idx = static_cast<const IntStorage *>(indices.storage.get());
  auto *W = static_cast<const T *>(weight.storage.get());
  auto *O = static_cast<T *>(out.storage.get());

  const tcapint I_s = indices.stride[0];
  const tcapint W_s0 = weight.stride[0];
  const tcapint W_s1 = weight.stride[1];
  const tcapint O_s = out.stride.back();

  pfControl.par_for(0U, N, [&](const tcapint &i, const unsigned &) {
    const tcapint token = (*idx)[indices.offset + i * I_s];

    const tcapint w_base = weight.offset + token * W_s0;
    const tcapint o_base = i * O_s;

    for (tcapint d = 0U; d < D; ++d) {
      O->write(o_base + d * O_s, (*W)[w_base + d * W_s1]);
    }
  });
}

template <typename T1, typename T2>
void cpu_backward(Tensor &dW, const SymbolTensor &indices, const Tensor &dout) {
  const tcapint D = dW.shape[1];
  const size_t N = indices.get_broadcast_size();

  auto *idx = static_cast<const IntStorage *>(indices.storage.get());
  auto *dWt = static_cast<T1 *>(dW.storage.get());
  auto *dOut = static_cast<const T2 *>(dout.storage.get());

  const tcapint I_s = indices.stride[0];
  const tcapint W_s0 = dW.stride[0];
  const tcapint W_s1 = dW.stride[1];
  const tcapint O_s = dout.stride.back();

  pfControl.par_for(0U, N, [&](const tcapint &i, const unsigned &) {
    const tcapint token = (*idx)[indices.offset + i * I_s];

    const tcapint w_base = dW.offset + token * W_s0;
    const tcapint o_base = dout.offset + i * O_s;

    for (tcapint d = 0U; d < D; ++d) {
      dWt->add(w_base + d * W_s1, (*dOut)[o_base + d * O_s]);
    }
  });
}

#if ENABLE_GPU
void gpu_forward_real(const SymbolTensor &indices, const Tensor &weight,
                      Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuIntStorage, GpuRealStorage, GpuRealStorage,
                      OCL_API_EMBEDDING_REAL);
}
void gpu_forward_complex(const SymbolTensor &indices, const Tensor &weight,
                         Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuIntStorage, GpuComplexStorage, GpuComplexStorage,
                      OCL_API_EMBEDDING_COMPLEX);
}
void gpu_backward_real(Tensor &out, const SymbolTensor &indices,
                       const Tensor &weight) {
  DISPATCH_GPU_KERNEL(GpuIntStorage, GpuRealStorage, GpuRealStorage,
                      OCL_API_EMBEDDING_GRAD_REAL);
}
void gpu_backward_complex(Tensor &out, const SymbolTensor &indices,
                          const Tensor &weight) {
  DISPATCH_GPU_KERNEL(GpuIntStorage, GpuRealStorage, GpuRealStorage,
                      OCL_API_EMBEDDING_GRAD_REAL);
}
void gpu_backward_mixed(Tensor &out, const SymbolTensor &indices,
                        const Tensor &weight) {
  DISPATCH_GPU_KERNEL(GpuIntStorage, GpuRealStorage, GpuRealStorage,
                      OCL_API_EMBEDDING_GRAD_REAL);
}
#endif

void embedding_gather(const SymbolTensor &a, const Tensor &b, Tensor &o) {
  validate_all_same_device({&a, &b, &o}, "embedding_gather");
  const bool isAComplex = a.storage->dtype == DType::COMPLEX;
  if (isAComplex) {
    DEVICE_SWITCH(cpu_forward<ComplexStorage>, gpu_forward_complex, a, b, o);
  } else {
    DEVICE_SWITCH(cpu_forward<RealStorage>, gpu_forward_real, a, b, o);
  }
}

void embedding_scatter_add(Tensor &o, const SymbolTensor &a, const Tensor &b) {
  validate_all_same_device({&a, &b, &o}, "embedding_gather");
  const bool isAComplex = a.storage->dtype == DType::COMPLEX;
  const bool isOComplex = o.storage->dtype == DType::COMPLEX;
  if (isAComplex && !isOComplex) {
    throw std::invalid_argument(
        "Cannot combine complex tensors into real1 tensor!");
  }
  if (isOComplex) {
    if (isAComplex) {
      if (o.storage->device == DeviceTag::GPU) {
        cpu_backward<ComplexStorage, ComplexStorage>(o, a, b);
      } else {
        gpu_backward_complex(o, a, b);
      }
    } else {
      if (o.storage->device == DeviceTag::GPU) {
        cpu_backward<ComplexStorage, RealStorage>(o, a, b);
      } else {
        gpu_backward_mixed(o, a, b);
      }
    }
  } else {
    if (o.storage->device == DeviceTag::GPU) {
      cpu_backward<RealStorage, RealStorage>(o, a, b);
    } else {
      gpu_backward_real(o, a, b);
    }
  }
}
} // namespace Weed
