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

#define REDUCE_KERNEL(type)                                                    \
  const tcapint n = out.get_broadcast_size();                                  \
  const int64_t id = index;                                                    \
  pfControl.par_for(0, n, [&](const tcapint &o, const unsigned &cpu) {         \
    tcapint base = 0U;                                                         \
    tcapint tmp = o;                                                           \
                                                                               \
    for (int64_t d = a.shape.size() - 1; d >= 0; --d) {                        \
      if (d == id) {                                                           \
        continue;                                                              \
      }                                                                        \
                                                                               \
      tcapint dim = a.shape[d];                                                \
      tcapint i_d = tmp % dim;                                                 \
      tmp /= dim;                                                              \
                                                                               \
      base += i_d * a.stride[d];                                               \
    }                                                                          \
                                                                               \
    type sum = ZERO_R1;                                                        \
    for (tcapint j = 0U; j < a.shape[id]; ++j) {                               \
      sum += (*pa)[base + j * a.stride[id]];                                   \
    }                                                                          \
    po->write(o, sum);                                                         \
  });

#define REDUCE_GRAD_KERNEL(type)                                               \
  const tcapint n = din.get_broadcast_size();                                  \
  const int64_t id = index;                                                    \
  pfControl.par_for(0, n, [&](const tcapint &i, const unsigned &cpu) {         \
    tcapint tmp = i;                                                           \
    tcapint o = 0U;                                                            \
                                                                               \
    for (int64_t d = in.shape.size() - 1; d >= 0; --d) {                       \
      if (d == id) {                                                           \
        continue;                                                              \
      }                                                                        \
                                                                               \
      tcapint dim = in.shape[d];                                               \
      tcapint i_d = tmp % dim;                                                 \
      tmp /= dim;                                                              \
                                                                               \
      o += i_d * dout.stride[d];                                               \
    }                                                                          \
    pdi->add(i, (type)(*po)[o]);                                               \
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

#define DEVICE_SWITCH_GRAD(cpu, gpu)                                           \
  switch (din.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(index, din, in, dout);                                                 \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(index, din, in, dout);                                                 \
  }

#define CPU_GRAD(t1, t2, type)                                                 \
  GET_FLAT_TENSOR(t1, din, pdi);                                               \
  GET_CONST_FLAT_TENSOR(t2, dout, po);                                         \
  REDUCE_GRAD_KERNEL(type);

#define GPU_GRAD_ARGS()                                                        \
  const tcapint args[10U] {                                                    \
    din.offset, din.stride[0U], in.offset, in.stride[0U], dout.offset,         \
        dout.stride[0U], 0U, 0U, 0U, 0U                                        \
  }

#define GPU_GRAD(type1, type2, api_call)                                       \
  GPU_GRAD_ARGS();                                                             \
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

namespace Weed {
void ReduceKernel::cpu_real(const tcapint &index, const Tensor &a,
                            Tensor &out) {
  GET_CONST_FLAT_TENSOR(RealTensor, a, pa);
  GET_STORAGE(RealStorage, out, po);

  REDUCE_KERNEL(real1);
}
void ReduceKernel::cpu_complex(const tcapint &index, const Tensor &a,
                               Tensor &out) {
  GET_CONST_FLAT_TENSOR(ComplexTensor, a, pa);
  GET_STORAGE(ComplexStorage, out, po);

  REDUCE_KERNEL(complex);
}
void ReduceKernel::cpu_grad_real(const tcapint &index, Tensor &din,
                                 const Tensor &in, const Tensor &dout) {
  CPU_GRAD(RealTensor, RealTensor, real1);
}
void ReduceKernel::cpu_grad_complex(const tcapint &index, Tensor &din,
                                    const Tensor &in, const Tensor &dout) {
  CPU_GRAD(ComplexTensor, ComplexTensor, complex);
}
void ReduceKernel::cpu_grad_mixed(const tcapint &index, Tensor &din,
                                  const Tensor &in, const Tensor &dout) {
  CPU_GRAD(ComplexTensor, RealTensor, complex);
}

#if ENABLE_GPU
void ReduceKernel::gpu_real(const tcapint &index, const Tensor &a,
                            Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, OCL_API_REDUCE_REAL);
}
void ReduceKernel::gpu_complex(const tcapint &index, const Tensor &a,
                               Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, OCL_API_REDUCE_COMPLEX);
}
void ReduceKernel::gpu_grad_real(const tcapint &index, Tensor &din,
                                 const Tensor &in, const Tensor &dout) {
  GPU_GRAD(GpuRealStorage, GpuRealStorage, OCL_API_REDUCE_GRAD_REAL);
}
void ReduceKernel::gpu_grad_complex(const tcapint &index, Tensor &din,
                                    const Tensor &in, const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuComplexStorage, OCL_API_REDUCE_GRAD_COMPLEX);
}
void ReduceKernel::gpu_grad_mixed(const tcapint &index, Tensor &din,
                                  const Tensor &in, const Tensor &dout) {
  GPU_GRAD(GpuRealStorage, GpuComplexStorage, OCL_API_REDUCE_GRAD_MIXED);
}
#endif

void ReduceKernel::reduce(const tcapint &index, const Tensor &a, Tensor &out) {
  validate_all_same_device({&a, &out}, "ReduceKernel::reduce");
  if (a.storage->dtype != out.storage->dtype) {
    throw std::invalid_argument("Output tensor dtype mismatch!");
  }
  if (a.storage->dtype == DType::COMPLEX) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_complex, gpu_complex);
#else
    cpu_complex(index, a, out);
#endif
  } else {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_real, gpu_real);
#else
    cpu_real(index, a, out);
#endif
  }
}

void ReduceKernel::reduce_grad(const tcapint &index, Tensor &din,
                               const Tensor &in, const Tensor &dout) {
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
      DEVICE_SWITCH_GRAD(cpu_grad_complex, gpu_grad_complex);
#else
      cpu_grad_complex(index, din, in, dout);
#endif
      break;
    case DType::REAL:
    default:
#if ENABLE_GPU
      DEVICE_SWITCH_GRAD(cpu_grad_mixed, gpu_grad_mixed);
#else
      cpu_grad_complex(index, din, in, dout);
#endif
    }
    break;
  case DType::REAL:
  default:
#if ENABLE_GPU
    DEVICE_SWITCH_GRAD(cpu_grad_real, gpu_grad_real);
#else
    cpu_grad_real(index, din, in, dout);
#endif
  }
}

ReduceKernel reduce_kernel;

void reduce(const tcapint &index, const Tensor &a, Tensor &out) {
  reduce_kernel.reduce(index, a, out);
}
void reduce_grad(const tcapint &index, Tensor &din, const Tensor &a,
                 const Tensor &dout) {
  reduce_kernel.reduce_grad(index, din, a, dout);
}
} // namespace Weed
