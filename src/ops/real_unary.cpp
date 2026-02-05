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

#include "ops/real_unary.hpp"
#include "common/parallel_for.hpp"
#include "ops/util.hpp"
#include "tensors/flat_tensors.hpp"

#include <iostream>

#define GPU_GRAD(type1, type2, type3, api_call)                                \
  GPU_GRAD_ARGS();                                                             \
  std::shared_ptr<type1> a_storage =                                           \
      std::dynamic_pointer_cast<type1>(din.storage);                           \
  std::shared_ptr<type2> b_storage =                                           \
      std::dynamic_pointer_cast<type2>(in.storage);                            \
  std::shared_ptr<type3> c_storage =                                           \
      std::dynamic_pointer_cast<type3>(dout.storage);                          \
  a_storage->dev->RequestKernel(                                               \
      OCLAPI::api_call, args, din.get_size(),                                  \
      {a_storage->buffer, b_storage->buffer, c_storage->buffer})

#define DEVICE_SWITCH(cpu, gpu, a, b, out)                                     \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, b, out);                                                            \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, b, out);                                                            \
  }

#define GPU_GRAD_ARGS()                                                        \
  const tcapint args[10U] {                                                    \
    din.offset, din.stride[0U], in.offset, in.stride[0U], dout.offset,         \
        dout.stride[0U], 0U, 0U, 0U, 0U                                        \
  }

#define CPU_RELU_GRAD()                                                        \
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    if ((*pi)[i] > 0) {                                                        \
      pdi->add(i, (*po)[i]);                                                   \
    }                                                                          \
  }

#define CPU_SIGMOID_GRAD()                                                     \
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    const real1 yi = (*pi)[i];                                                 \
    pdi->add(i, yi *(ONE_R1 - yi) * (*po)[i]);                                 \
  }

#define CPU_TANH_GRAD()                                                        \
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    const real1 y = (*pi)[i];                                                  \
    pdi->add(i, (*po)[i] * (ONE_R1 - y * y));                                  \
  }

namespace Weed {
static void cpu_relu(const Tensor &a, Tensor &out) {
  CPU_INIT_2(RealTensor, RealTensor);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    po->write(i, std::max((*pa)[i], ZERO_R1));
  };
  SPARSE_CPU_2_RUN(SparseCpuRealStorage);
}
template <typename T1, typename T2>
static void cpu_relu_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
  CPU_GRAD_INIT_3(T1, RealTensor, T2);
  CPU_RELU_GRAD();
  SPARSE_CPU_GRAD_3_RUN(SparseCpuRealStorage, SparseCpuRealStorage);
}
static inline void cpu_relu_grad_real(Tensor &din, const Tensor &in,
                                      const Tensor &dout) {
  cpu_relu_grad<RealTensor, RealTensor>(din, in, dout);
}
static inline void cpu_relu_grad_complex(Tensor &din, const Tensor &in,
                                         const Tensor &dout) {
  cpu_relu_grad<ComplexTensor, ComplexTensor>(din, in, dout);
}
static inline void cpu_relu_grad_mixed(Tensor &din, const Tensor &in,
                                       const Tensor &dout) {
  cpu_relu_grad<ComplexTensor, RealTensor>(din, in, dout);
}

#if ENABLE_GPU
static void gpu_relu(const Tensor &a, Tensor &out) {
  const tcapint args[10U]{
      a.offset, a.stride[0U], out.offset, out.stride[0U], 0U, 0U, 0U,
      0U,       0U,           0U};
  GpuRealStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(a.storage);
  GpuRealStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(out.storage);
  a_storage->dev->RequestKernel(OCLAPI::OCL_API_RELU, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer});
}
static void gpu_relu_grad_real(Tensor &din, const Tensor &in,
                               const Tensor &dout) {
  GPU_GRAD(GpuRealStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_RELU_GRAD_REAL);
}
static void gpu_relu_grad_complex(Tensor &din, const Tensor &in,
                                  const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuRealStorage, GpuComplexStorage,
           OCL_API_RELU_GRAD_COMPLEX);
}
static void gpu_relu_grad_mixed(Tensor &din, const Tensor &in,
                                const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_RELU_GRAD_MIXED);
}
#endif

static void cpu_sigmoid(const Tensor &a, Tensor &out) {
  CPU_INIT_2(RealTensor, RealTensor);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    po->write(i, ONE_R1 / (ONE_R1 + exp(-(*pa)[i])));
  };
  SPARSE_CPU_2_RUN(SparseCpuRealStorage);
}
template <typename T1, typename T2>
static void cpu_sigmoid_grad(Tensor &din, const Tensor &in,
                             const Tensor &dout) {
  CPU_GRAD_INIT_3(T1, RealTensor, T2);
  CPU_SIGMOID_GRAD();
  SPARSE_CPU_GRAD_3_RUN(SparseCpuRealStorage, SparseCpuRealStorage);
}
static inline void cpu_sigmoid_grad_real(Tensor &din, const Tensor &in,
                                         const Tensor &dout) {
  cpu_sigmoid_grad<RealTensor, RealTensor>(din, in, dout);
}
static inline void cpu_sigmoid_grad_complex(Tensor &din, const Tensor &in,
                                            const Tensor &dout) {
  cpu_sigmoid_grad<ComplexTensor, ComplexTensor>(din, in, dout);
}
static inline void cpu_sigmoid_grad_mixed(Tensor &din, const Tensor &in,
                                          const Tensor &dout) {
  cpu_sigmoid_grad<ComplexTensor, RealTensor>(din, in, dout);
}

#if ENABLE_GPU
static void gpu_sigmoid(const Tensor &a, Tensor &out) {
  const tcapint args[10U]{
      a.offset, a.stride[0U], out.offset, out.stride[0U], 0U, 0U, 0U,
      0U,       0U,           0U};
  GpuRealStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(a.storage);
  GpuRealStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(out.storage);
  a_storage->dev->RequestKernel(OCLAPI::OCL_API_SIGMOID, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer});
}
static void gpu_sigmoid_grad_real(Tensor &din, const Tensor &in,
                                  const Tensor &dout) {
  GPU_GRAD(GpuRealStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_SIGMOID_GRAD_REAL);
}
static void gpu_sigmoid_grad_complex(Tensor &din, const Tensor &in,
                                     const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuRealStorage, GpuComplexStorage,
           OCL_API_SIGMOID_GRAD_COMPLEX);
}
static void gpu_sigmoid_grad_mixed(Tensor &din, const Tensor &in,
                                   const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_SIGMOID_GRAD_MIXED);
}
#endif

static void cpu_tanh(const Tensor &a, Tensor &out) {
  CPU_INIT_2(RealTensor, RealTensor);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    po->write(i, (real1)std::tanh((*pa)[i]));
  };
  SPARSE_CPU_2_RUN(SparseCpuRealStorage);
}
template <typename T1, typename T2>
static void cpu_tanh_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
  CPU_GRAD_INIT_3(T1, RealTensor, T2);
  CPU_TANH_GRAD();
  SPARSE_CPU_GRAD_3_RUN(SparseCpuRealStorage, SparseCpuRealStorage);
}
static inline void cpu_tanh_grad_real(Tensor &din, const Tensor &in,
                                      const Tensor &dout) {
  cpu_tanh_grad<RealTensor, RealTensor>(din, in, dout);
}
static inline void cpu_tanh_grad_complex(Tensor &din, const Tensor &in,
                                         const Tensor &dout) {
  cpu_tanh_grad<ComplexTensor, ComplexTensor>(din, in, dout);
}
static inline void cpu_tanh_grad_mixed(Tensor &din, const Tensor &in,
                                       const Tensor &dout) {
  cpu_tanh_grad<ComplexTensor, RealTensor>(din, in, dout);
}

#if ENABLE_GPU
static void gpu_tanh(const Tensor &a, Tensor &out) {
  const tcapint args[10U]{
      a.offset, a.stride[0U], out.offset, out.stride[0U], 0U, 0U, 0U,
      0U,       0U,           0U};
  GpuRealStoragePtr a_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(a.storage);
  GpuRealStoragePtr o_storage =
      std::dynamic_pointer_cast<GpuRealStorage>(out.storage);
  a_storage->dev->RequestKernel(OCLAPI::OCL_API_TANH, args, a.get_size(),
                                {a_storage->buffer, o_storage->buffer});
}
static void gpu_tanh_grad_real(Tensor &din, const Tensor &in,
                               const Tensor &dout) {
  GPU_GRAD(GpuRealStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_TANH_GRAD_REAL);
}
static void gpu_tanh_grad_complex(Tensor &din, const Tensor &in,
                                  const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuRealStorage, GpuComplexStorage,
           OCL_API_TANH_GRAD_COMPLEX);
}
static void gpu_tanh_grad_mixed(Tensor &din, const Tensor &in,
                                const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_TANH_GRAD_MIXED);
}
#endif

void RealUnaryKernel::unary(const Tensor &a, Tensor &out) {
  validate_all_same_device({&a, &out}, "RealUnaryKernel::unary");
  if (a.get_broadcast_size() != out.get_broadcast_size()) {
    throw std::invalid_argument(
        "In Weed::unary(a, out), out size does not match input size!");
  }
  if ((a.storage->dtype == DType::COMPLEX) ||
      (out.storage->dtype == DType::COMPLEX)) {
    throw std::invalid_argument(
        "Cannot apply RealUnary activation functions on complex tensors!");
  }
#if ENABLE_GPU
  switch (out.storage->device) {
  case DeviceTag::GPU:
    gpu(a, out);
    break;
  case DeviceTag::CPU:
  default:
    cpu(a, out);
  }
#else
  cpu(a, out);
#endif
}

void RealUnaryKernel::unary_grad(Tensor &din, const Tensor &in,
                                 const Tensor &dout) {
  validate_all_same_device({&din, &in, &dout}, "RealUnaryKernel::unary_grad");
  if ((din.storage->dtype == DType::REAL) &&
      (dout.storage->dtype != DType::REAL)) {
    throw std::invalid_argument(
        "In Weed::unary_grad(din, in, dout), dout dtype "
        "must upcast to dout dtype!");
  }
  const tcapint dinSize = din.get_broadcast_size();
  const tcapint inSize = in.get_broadcast_size();
  const tcapint doutSize = dout.get_broadcast_size();
  if ((dinSize != inSize) || (dinSize != doutSize)) {
    std::cout << dinSize << " " << inSize << " " << doutSize << std::endl;
    throw std::invalid_argument(
        "In Weed::unary_grad(din, in, dout), sizes do not match!");
  }
  if (in.storage->dtype != DType::REAL) {
    throw std::invalid_argument(
        "In Weed::unary_grad(din, in, dout), 'in' dtype must be real-number!");
  }
  switch (din.storage->dtype) {
  case DType::COMPLEX:
    switch (dout.storage->dtype) {
    case DType::COMPLEX:
#if ENABLE_GPU
      DEVICE_SWITCH(cpu_grad_complex, gpu_grad_complex, din, in, dout);
#else
      cpu_grad_complex(din, in, dout);
#endif
      break;
    case DType::REAL:
    default:
#if ENABLE_GPU
      DEVICE_SWITCH(cpu_grad_mixed, gpu_grad_mixed, din, in, dout);
#else
      cpu_grad_complex(din, in, dout);
#endif
    }
    break;
  case DType::REAL:
  default:
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_grad_real, gpu_grad_real, din, in, dout);
#else
    cpu_grad_real(din, in, dout);
#endif
  }
}

RealUnaryKernel relu_kernel = {cpu_relu,
                               cpu_relu_grad_real,
                               cpu_relu_grad_complex,
                               cpu_relu_grad_mixed,
#if ENABLE_GPU
                               gpu_relu,
                               gpu_relu_grad_real,
                               gpu_relu_grad_complex,
                               gpu_relu_grad_mixed
#endif
};

RealUnaryKernel sigmoid_kernel = {cpu_sigmoid,
                                  cpu_sigmoid_grad_real,
                                  cpu_sigmoid_grad_complex,
                                  cpu_sigmoid_grad_mixed,
#if ENABLE_GPU
                                  gpu_sigmoid,
                                  gpu_sigmoid_grad_real,
                                  gpu_sigmoid_grad_complex,
                                  gpu_sigmoid_grad_mixed
#endif
};

RealUnaryKernel tanh_kernel = {cpu_tanh,
                               cpu_tanh_grad_real,
                               cpu_tanh_grad_complex,
                               cpu_tanh_grad_mixed,
#if ENABLE_GPU
                               gpu_tanh,
                               gpu_tanh_grad_real,
                               gpu_tanh_grad_complex,
                               gpu_tanh_grad_mixed
#endif
};

void relu(const Tensor &a, Tensor &out) { relu_kernel.unary(a, out); }
void relu_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
  relu_kernel.unary_grad(din, in, dout);
}

void sigmoid(const Tensor &a, Tensor &out) { sigmoid_kernel.unary(a, out); }
void sigmoid_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
  sigmoid_kernel.unary_grad(din, in, dout);
}

void tanh(const Tensor &a, Tensor &out) { tanh_kernel.unary(a, out); }
void tanh_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
  tanh_kernel.unary_grad(din, in, dout);
}
} // namespace Weed
