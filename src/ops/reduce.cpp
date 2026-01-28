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
#include "storage/all_storage.hpp"

#define REDUCE_KERNEL(type)                                                    \
  const tcapint I_o = out.stride[0U];                                          \
  const size_t n = out.get_size();                                             \
  const int64_t id = index;                                                    \
  pfControl.par_for(0, n, [&](const tcapint &o, const unsigned &cpu) {         \
    tcapint base = a.offset;                                                   \
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
      sum += pa[base + j * a.stride[id]];                                      \
    }                                                                          \
    po.write(o *I_o, sum);                                                     \
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

namespace Weed {
void ReduceKernel::cpu_real(const size_t &index, const Tensor &a, Tensor &out) {
  GET_STORAGE(CpuRealStorage, a, pa);
  GET_STORAGE(CpuRealStorage, out, po);

  REDUCE_KERNEL(real1);
}
void ReduceKernel::cpu_complex(const size_t &index, const Tensor &a,
                               Tensor &out) {
  GET_STORAGE(CpuComplexStorage, a, pa);
  GET_STORAGE(CpuComplexStorage, out, po);

  REDUCE_KERNEL(complex);
}

#if ENABLE_GPU
void ReduceKernel::gpu_real(const size_t &index, const Tensor &a, Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuRealStorage, OCL_API_REDUCE_REAL);
}
void ReduceKernel::gpu_complex(const size_t &index, const Tensor &a,
                               Tensor &out) {
  DISPATCH_GPU_KERNEL(GpuComplexStorage, OCL_API_REDUCE_COMPLEX);
}
#endif

void ReduceKernel::reduce(const size_t &index, const Tensor &a, Tensor &out) {
  if (a.storage->dtype != out.storage->dtype) {
    throw std::invalid_argument("Output tensor dtype mismatch!");
  }
  if (a.get_broadcast_size() != out.get_broadcast_size()) {
    throw std::invalid_argument(
        "In Weed::reduce(a, out), out size does not match input size!");
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

ReduceKernel reduce_kernel;

void reduce(const size_t &index, const Tensor &a, Tensor &out) {
  reduce_kernel.reduce(index, a, out);
}
} // namespace Weed
