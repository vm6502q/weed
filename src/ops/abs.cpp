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

#include "ops/abs.hpp"
#include "common/parallel_for.hpp"
#include "ops/util.hpp"
#include "tensors/flat_tensors.hpp"

#define GPU(type1, type2, api_call)                                            \
  GPU_ARGS();                                                                  \
  std::shared_ptr<type1> a_storage =                                           \
      std::dynamic_pointer_cast<type1>(a.storage);                             \
  std::shared_ptr<type2> o_storage =                                           \
      std::dynamic_pointer_cast<type2>(out.storage);                           \
  a_storage->dev->RequestKernel(OCLAPI::api_call, args, a.get_size(),          \
                                {a_storage->buffer, o_storage->buffer})

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

#define DEVICE_SWITCH(cpu, gpu, a, out)                                        \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, out);                                                               \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, out);                                                               \
  }

#define DEVICE_SWITCH_GRAD(cpu, gpu, a, b, out)                                \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, b, out);                                                            \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, b, out);                                                            \
  }

#define GPU_ARGS()                                                             \
  const tcapint args[10U] {                                                    \
    a.offset, a.stride[0U], out.offset, out.stride[0U], 0U, 0U, 0U, 0U, 0U, 0U \
  }

#define GPU_GRAD_ARGS()                                                        \
  const tcapint args[10U] {                                                    \
    din.offset, din.stride[0U], in.offset, in.stride[0U], dout.offset,         \
        dout.stride[0U], 0U, 0U, 0U, 0U                                        \
  }

#define REAL_ABS_GRAD_KERNEL()                                                 \
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    const real1 tmp = (*pi)[i];                                                \
    if (tmp != ZERO_R1) {                                                      \
      const real1 tmp_o = (*po)[i];                                            \
      pdi->add(i, (tmp > ZERO_R1) ? tmp_o : -tmp_o);                           \
    }                                                                          \
  }

#define COMPLEX_ABS_GRAD_KERNEL()                                              \
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    const complex tmp = (*pi)[i];                                              \
    if (tmp != ZERO_CMPLX) {                                                   \
      pdi->add(i, tmp *((*po)[i] / std::abs(tmp)));                            \
    }                                                                          \
  }

namespace Weed {
static inline void cpu_real(const Tensor &a, Tensor &out) {
  CPU_INIT_2(RealTensor, RealTensor);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    const real1 tmp = (*pa)[i];
    po->write(i, (tmp < ZERO_R1) ? -tmp : tmp);
  };
  SPARSE_CPU_2_RUN(SparseCpuRealStorage);
}
static inline void cpu_complex(const Tensor &a, Tensor &out) {
  CPU_INIT_2(ComplexTensor, RealTensor);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    po->write(i, (real1)std::abs((*pa)[i]));
  };
  SPARSE_CPU_2_RUN(SparseCpuComplexStorage);
}
#if ENABLE_GPU
static inline void gpu_real(const Tensor &a, Tensor &out) {
  GPU(GpuRealStorage, GpuRealStorage, OCL_API_ABS_REAL);
}
static inline void gpu_complex(const Tensor &a, Tensor &out) {
  GPU(GpuComplexStorage, GpuRealStorage, OCL_API_ABS_COMPLEX);
}
#endif
void abs(const Tensor &a, Tensor &out) {
  validate_all_same_device({&a, &out}, "AbsKernel::abs");
  if (a.get_size() != out.get_size()) {
    throw std::invalid_argument(
        "In Weed::abs(a, out), out size does not match input size!");
  }
  switch (a.storage->dtype) {
  case DType::COMPLEX:
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_complex, gpu_complex, a, out)
#else
    cpu_complex(a, out);
#endif
    break;
  case DType::REAL:
  default:
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_real, gpu_real, a, out)
#else
    cpu_real(a, out);
#endif
  }
}

template <typename T1, typename T2>
static void cpu_real_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
  CPU_GRAD_INIT_3(T1, RealTensor, RealTensor);
  REAL_ABS_GRAD_KERNEL();
  SPARSE_CPU_GRAD_3_RUN(T2, SparseCpuRealStorage);
}
template <typename T1, typename T2, typename T3>
static void cpu_complex_grad(Tensor &din, const Tensor &in,
                             const Tensor &dout) {
  CPU_GRAD_INIT_3(ComplexTensor, T1, T2);
  COMPLEX_ABS_GRAD_KERNEL();
  SPARSE_CPU_GRAD_3_RUN(SparseCpuComplexStorage, T3);
}
static inline void cpu_real_grad_real(Tensor &din, const Tensor &in,
                                      const Tensor &dout) {
  cpu_real_grad<RealTensor, SparseCpuRealStorage>(din, in, dout);
}
static inline void cpu_real_grad_complex(Tensor &din, const Tensor &in,
                                         const Tensor &dout) {
  cpu_complex_grad<RealTensor, ComplexTensor, SparseCpuComplexStorage>(din, in,
                                                                       dout);
}
static inline void cpu_real_grad_mixed(Tensor &din, const Tensor &in,
                                       const Tensor &dout) {
  cpu_real_grad<ComplexTensor, SparseCpuComplexStorage>(din, in, dout);
}
static inline void cpu_complex_grad_real(Tensor &din, const Tensor &in,
                                         const Tensor &dout) {
  cpu_complex_grad<ComplexTensor, RealTensor, SparseCpuRealStorage>(din, in,
                                                                    dout);
}
static inline void cpu_complex_grad_complex(Tensor &din, const Tensor &in,
                                            const Tensor &dout) {
  cpu_complex_grad<ComplexTensor, ComplexTensor, SparseCpuComplexStorage>(
      din, in, dout);
}
static inline void cpu_complex_grad_mixed(Tensor &din, const Tensor &in,
                                          const Tensor &dout) {
  cpu_complex_grad<ComplexTensor, RealTensor, SparseCpuRealStorage>(din, in,
                                                                    dout);
}
#if ENABLE_GPU
static inline void gpu_real_grad_real(Tensor &din, const Tensor &in,
                                      const Tensor &dout) {
  GPU_GRAD(GpuRealStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_ABS_REAL_GRAD_REAL);
}
static inline void gpu_real_grad_complex(Tensor &din, const Tensor &in,
                                         const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuRealStorage, GpuComplexStorage,
           OCL_API_ABS_REAL_GRAD_COMPLEX);
}
static inline void gpu_real_grad_mixed(Tensor &din, const Tensor &in,
                                       const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_ABS_REAL_GRAD_MIXED);
}
static inline void gpu_complex_grad_real(Tensor &din, const Tensor &in,
                                         const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuComplexStorage, GpuRealStorage,
           OCL_API_ABS_COMPLEX_GRAD_REAL);
}
static inline void gpu_complex_grad_complex(Tensor &din, const Tensor &in,
                                            const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuComplexStorage, GpuComplexStorage,
           OCL_API_ABS_COMPLEX_GRAD_COMPLEX);
}
static inline void gpu_complex_grad_mixed(Tensor &din, const Tensor &in,
                                          const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuComplexStorage, GpuRealStorage,
           OCL_API_ABS_COMPLEX_GRAD_MIXED);
}
#endif
void abs_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
  validate_all_same_device({&din, &in, &dout}, "AbsKernel::abs_grad");
  if ((din.storage->dtype == DType::REAL) &&
      (dout.storage->dtype != DType::REAL)) {
    throw std::invalid_argument("In Weed::abs_grad(din, in, dout), dout dtype "
                                "must upcast to dout dtype!");
  }
  const size_t dinSize = din.get_size();
  const size_t inSize = in.get_size();
  const size_t doutSize = dout.get_size();
  if ((dinSize != inSize) || (dinSize != doutSize)) {
    throw std::invalid_argument(
        "In Weed::abs_grad(din, in, dout), sizes do not match!");
  }
  switch (in.storage->dtype) {
  case DType::COMPLEX:
    switch (din.storage->dtype) {
    case DType::COMPLEX:
      switch (dout.storage->dtype) {
      case DType::COMPLEX:
#if ENABLE_GPU
        DEVICE_SWITCH_GRAD(cpu_complex_grad_complex, gpu_complex_grad_complex,
                           din, in, dout)
#else
        cpu_complex_grad_complex(din, in, dout);
#endif
        break;
      case DType::REAL:
      default:
#if ENABLE_GPU
        DEVICE_SWITCH_GRAD(cpu_complex_grad_mixed, gpu_complex_grad_mixed, din,
                           in, dout)
#else
        cpu_complex_grad_mixed(din, in, dout);
#endif
      }
      break;
    case DType::REAL:
    default:
#if ENABLE_GPU
      DEVICE_SWITCH_GRAD(cpu_complex_grad_real, gpu_complex_grad_real, din, in,
                         dout);
#else
      cpu_complex_grad_real(din, in, dout);
#endif
    }
    break;
  case DType::REAL:
  default:
    switch (din.storage->dtype) {
    case DType::COMPLEX:
      switch (dout.storage->dtype) {
      case DType::COMPLEX:
#if ENABLE_GPU
        DEVICE_SWITCH_GRAD(cpu_real_grad_complex, gpu_real_grad_complex, din,
                           in, dout);
#else
        cpu_real_grad_complex(din, in, dout);
#endif
        break;
      case DType::REAL:
      default:
#if ENABLE_GPU
        DEVICE_SWITCH_GRAD(cpu_real_grad_mixed, gpu_real_grad_mixed, din, in,
                           dout)
#else
        cpu_real_grad_mixed(din, in, dout);
#endif
      }
      break;
    case DType::REAL:
    default:
#if ENABLE_GPU
      DEVICE_SWITCH_GRAD(cpu_real_grad_real, gpu_real_grad_real, din, in, dout);
#else
      cpu_real_grad_real(din, in, dout);
#endif
    }
  }
}
} // namespace Weed
