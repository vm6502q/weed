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
#include "storage/all_storage.hpp"

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
namespace Weed {
void AbsKernel::cpu_real(const Tensor &a, Tensor &out) {
  CPU_INIT_2(RealStorage, RealStorage);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    real1 tmp = (*pa)[O_a + i * I_a];
    po->write(i * I_o, (tmp < ZERO_R1) ? -tmp : tmp);
  };
  SPARSE_CPU_RUN(SparseCpuRealStorage);
}
void AbsKernel::cpu_complex(const Tensor &a, Tensor &out) {
  CPU_INIT_2(ComplexStorage, RealStorage);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    po->write(i * I_o, (real1)std::abs((*pa)[O_a + i * I_a]));
  };
  SPARSE_CPU_RUN(SparseCpuRealStorage);
}
#if ENABLE_GPU
void AbsKernel::gpu_real(const Tensor &a, Tensor &out) {
  GPU(GpuRealStorage, GpuRealStorage, OCL_API_ABS_REAL);
}
void AbsKernel::gpu_complex(const Tensor &a, Tensor &out) {
  GPU(GpuComplexStorage, GpuRealStorage, OCL_API_ABS_COMPLEX);
}
#endif
void AbsKernel::abs(const Tensor &a, Tensor &out) {
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

void AbsKernel::cpu_real_grad_real(Tensor &din, const Tensor &in,
                                   const Tensor &dout) {
  CPU_GRAD_INIT_3(RealStorage, RealStorage, RealStorage);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    const real1 tmp = (*pi)[O_i + i * I_i];
    if (tmp != ZERO_R1) {
      const real1 tmp_o = (*po)[O_o + i * I_o];
      pdi->add(O_d + i * I_d, (tmp > ZERO_R1) ? tmp_o : -tmp_o);
    }
  };
  SPARSE_CPU_GRAD_RUN(SparseCpuRealStorage, SparseCpuRealStorage);
}
void AbsKernel::cpu_real_grad_complex(Tensor &din, const Tensor &in,
                                      const Tensor &dout) {
  CPU_GRAD_INIT_3(ComplexStorage, RealStorage, ComplexStorage);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    const real1 tmp = (*pi)[O_i + i * I_i];
    if (tmp != ZERO_R1) {
      const complex tmp_o = (*po)[O_o + i * I_o];
      pdi->add(O_d + i * I_d, (tmp > ZERO_R1) ? tmp_o : -tmp_o);
    }
  };
  SPARSE_CPU_GRAD_RUN(SparseCpuComplexStorage, SparseCpuComplexStorage);
}
void AbsKernel::cpu_real_grad_mixed(Tensor &din, const Tensor &in,
                                    const Tensor &dout) {
  CPU_GRAD_INIT_3(ComplexStorage, RealStorage, RealStorage);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    const real1 tmp = (*pi)[O_i + i * I_i];
    if (tmp != ZERO_R1) {
      const real1 tmp_o = (*po)[O_o + i * I_o];
      pdi->add(O_d + i * I_d, (tmp > ZERO_R1) ? tmp_o : -tmp_o);
    }
  };
  SPARSE_CPU_GRAD_RUN(SparseCpuComplexStorage, SparseCpuRealStorage);
}
void AbsKernel::cpu_complex_grad_real(Tensor &din, const Tensor &in,
                                      const Tensor &dout) {
  CPU_GRAD_INIT_3(ComplexStorage, ComplexStorage, RealStorage);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    const complex tmp = (*pi)[O_i + i * I_i];
    if (tmp != ZERO_CMPLX) {
      pdi->add(O_d + i * I_d, tmp * ((*po)[O_o + i * I_o] / std::abs(tmp)));
    }
  };
  SPARSE_CPU_GRAD_RUN(SparseCpuComplexStorage, SparseCpuRealStorage);
}
void AbsKernel::cpu_complex_grad_complex(Tensor &din, const Tensor &in,
                                         const Tensor &dout) {
  CPU_GRAD_INIT_3(ComplexStorage, ComplexStorage, ComplexStorage);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    const complex tmp = (*pi)[O_i + i * I_i];
    if (tmp != ZERO_CMPLX) {
      pdi->add(O_d + i * I_d, (*po)[O_o + i * I_o] * tmp / std::abs(tmp));
    }
  };
  SPARSE_CPU_GRAD_RUN(SparseCpuComplexStorage, SparseCpuComplexStorage);
}
void AbsKernel::cpu_complex_grad_mixed(Tensor &din, const Tensor &in,
                                       const Tensor &dout) {
  CPU_GRAD_INIT_3(ComplexStorage, ComplexStorage, RealStorage);
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {
    const complex tmp = (*pi)[O_i + i * I_i];
    if (tmp != ZERO_CMPLX) {
      pdi->add(O_d + i * I_d, (*po)[O_o + i * I_o] * tmp / std::abs(tmp));
    }
  };
  SPARSE_CPU_GRAD_RUN(SparseCpuComplexStorage, SparseCpuRealStorage);
}
#if ENABLE_GPU
void AbsKernel::gpu_real_grad_real(Tensor &din, const Tensor &in,
                                   const Tensor &dout) {
  GPU_GRAD(GpuRealStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_ABS_REAL_GRAD_REAL);
}
void AbsKernel::gpu_real_grad_complex(Tensor &din, const Tensor &in,
                                      const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuRealStorage, GpuComplexStorage,
           OCL_API_ABS_REAL_GRAD_COMPLEX);
}
void AbsKernel::gpu_real_grad_mixed(Tensor &din, const Tensor &in,
                                    const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_ABS_REAL_GRAD_MIXED);
}
void AbsKernel::gpu_complex_grad_real(Tensor &din, const Tensor &in,
                                      const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuComplexStorage, GpuRealStorage,
           OCL_API_ABS_COMPLEX_GRAD_REAL);
}
void AbsKernel::gpu_complex_grad_complex(Tensor &din, const Tensor &in,
                                         const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuComplexStorage, GpuComplexStorage,
           OCL_API_ABS_COMPLEX_GRAD_COMPLEX);
}
void AbsKernel::gpu_complex_grad_mixed(Tensor &din, const Tensor &in,
                                       const Tensor &dout) {
  GPU_GRAD(GpuComplexStorage, GpuComplexStorage, GpuRealStorage,
           OCL_API_ABS_COMPLEX_GRAD_MIXED);
}
#endif
void AbsKernel::abs_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
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

AbsKernel abs_kernel;

void abs(const Tensor &a, Tensor &out) { abs_kernel.abs(a, out); }
void abs_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
  abs_kernel.abs_grad(din, in, dout);
}
} // namespace Weed
