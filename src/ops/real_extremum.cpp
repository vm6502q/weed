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

#include "ops/real_extremum.hpp"
#include "common/parallel_for.hpp"
#include "ops/util.hpp"
#include "tensors/flat_tensors.hpp"
#include "tensors/real_scalar.hpp"

#if ENABLE_GPU
#include "ops/gpu_flat.hpp"
#endif

#define CPU_INIT_1(ft)                                                         \
  GET_CONST_FLAT_TENSOR(ft, a, pa);                                            \
  const size_t n = a.get_broadcast_size()

#define SPARSE_CPU_1_RUN()                                                     \
  if (a.storage->is_sparse() && a.is_contiguous()) {                           \
    GET_STORAGE(SparseCpuRealStorage, a, sa);                                  \
    pfControl.par_for(sa->data, fn);                                           \
  } else {                                                                     \
    pfControl.par_for(1U, n, fn);                                              \
  }

#define GPU_GRAD(type1, type2, type3, api_call)                                \
  const real1 m = static_cast<const RealScalar *>(&out)->get_item();           \
  const complex v = complex(m);                                                \
  GPU_FLAT_GRAD_C(type1, type2, type3, OCLAPI::api_call, &v)

#define DEVICE_SWITCH(cpu, gpu, din, in, dout, out)                            \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(din, in, dout, out);                                                   \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(din, in, dout, out);                                                   \
  }


#define CPU_GRAD()                                                             \
  const real1 m = static_cast<const RealScalar *>(&out)->get_item();           \
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    if ((*pi)[i] == m) {                                                       \
      pdi->add(i, (*po)[i]);                                                   \
    }                                                                          \
  };                                                                           \
  pfControl.par_for(0, n, fn)

#define CPU_RUN_HEADER()                                                       \
  const unsigned cpuCount =                                                    \
      (unsigned)std::min(n, (size_t)pfControl.GetNumCores());                  \
  std::vector<real1> m(cpuCount, (*pa)[0U]);

#define CPU_MAX()                                                              \
  CPU_RUN_HEADER();                                                            \
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    const real1 v = (*pa)[i];                                                  \
    if (v > m[cpu]) {                                                          \
      m[cpu] = v;                                                              \
    }                                                                          \
  };                                                                           \
  SPARSE_CPU_1_RUN();                                                          \
  const real1 v = *std::max_element(m.begin(), m.end())

#define CPU_MIN()                                                              \
  CPU_RUN_HEADER();                                                            \
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    const real1 v = (*pa)[i];                                                  \
    if (v < m[cpu]) {                                                          \
      m[cpu] = v;                                                              \
    }                                                                          \
  };                                                                           \
  SPARSE_CPU_1_RUN();                                                          \
  const real1 v = *std::min_element(m.begin(), m.end());

namespace Weed {
static void cpu_max(const Tensor &a, Tensor &out) {
  CPU_INIT_2_SCALAR(RealTensor, RealTensor);
  CPU_MAX();
  po->write(0U, v);
}

static void cpu_min(const Tensor &a, Tensor &out) {
  CPU_INIT_2_SCALAR(RealTensor, RealTensor);
  CPU_MIN();
  po->write(0U, v);
}

template <typename T1, typename T2>
static void cpu_grad(Tensor &din, const Tensor &in, const Tensor &dout,
                     const Tensor &out) {
  CPU_GRAD_INIT_3(T1, RealTensor, T2);
  CPU_GRAD();
}
static inline void cpu_grad_real(Tensor &din, const Tensor &in,
                                 const Tensor &dout, const Tensor &out) {
  cpu_grad<RealTensor, RealTensor>(din, in, dout, out);
}
static inline void cpu_grad_complex(Tensor &din, const Tensor &in,
                                    const Tensor &dout, const Tensor &out) {
  cpu_grad<ComplexTensor, ComplexTensor>(din, in, dout, out);
}
static inline void cpu_grad_mixed(Tensor &din, const Tensor &in,
                                  const Tensor &dout, const Tensor &out) {
  cpu_grad<ComplexTensor, RealTensor>(din, in, dout, out);
}

#if ENABLE_GPU
// Copy the tensor to host memory in one read, reduce it with the CPU kernel,
// and write the single result back to the device.
template <typename Fn>
static void gpu_extremum(const Tensor &a, Tensor &out, Fn cpu_fn) {
  const TensorPtr ca = a.cast(DeviceTag::CPU);
  Tensor co(std::vector<real1>{ZERO_R1}, std::vector<tcapint>{1U}, false,
            DeviceTag::CPU);
  cpu_fn(*ca, co);
  const real1 v = (*static_cast<RealStorage *>(co.storage.get()))[0U];
  GpuRealStorage *o_storage = static_cast<GpuRealStorage *>(out.storage.get());
  o_storage->dev->SetReal(v, o_storage->buffer, out.offset);
}
static void gpu_max(const Tensor &a, Tensor &out) {
  gpu_extremum(a, out, cpu_max);
}
static void gpu_min(const Tensor &a, Tensor &out) {
  gpu_extremum(a, out, cpu_min);
}

static void gpu_grad_real(Tensor &din, const Tensor &in, const Tensor &dout,
                          const Tensor &out) {
  GPU_GRAD(GpuRealStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_MATCH_GRAD_REAL);
}
static void gpu_grad_complex(Tensor &din, const Tensor &in, const Tensor &dout,
                             const Tensor &out) {
  GPU_GRAD(GpuComplexStorage, GpuRealStorage, GpuComplexStorage,
           OCL_API_MATCH_GRAD_COMPLEX);
}
static void gpu_grad_mixed(Tensor &din, const Tensor &in, const Tensor &dout,
                           const Tensor &out) {
  GPU_GRAD(GpuComplexStorage, GpuRealStorage, GpuRealStorage,
           OCL_API_MATCH_GRAD_MIXED);
}
#endif

void RealExtremumKernel::extremum(const Tensor &a, Tensor &out) {
  validate_all_same_device({&a, &out}, "RealExtremumKernel::extremum");
  if ((a.storage->dtype == DType::COMPLEX) ||
      (out.storage->dtype == DType::COMPLEX)) {
    throw std::invalid_argument(
        "Cannot apply extremum reduction on complex tensors!");
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

void RealExtremumKernel::extremum_grad(Tensor &din, const Tensor &in,
                                       const Tensor &dout, const Tensor &out) {
  validate_all_same_device({&din, &in, &dout, &out},
                           "RealExtremumKernel::extremum_grad");
  if ((din.storage->dtype == DType::REAL) &&
      (dout.storage->dtype != DType::REAL)) {
    throw std::invalid_argument(
        "In RealExtremumKernel::extremum_grad(din, in, dout), dout dtype "
        "must upcast to dout dtype!");
  }
  if ((in.storage->dtype != DType::REAL) ||
      (out.storage->dtype != DType::REAL)) {
    throw std::invalid_argument(
        "In RealExtremumKernel::extremum_grad(din, in, dout), in and out dtype "
        "must be real-number!");
  }
  const tcapint dinSize = din.get_broadcast_size();
  const tcapint inSize = in.get_broadcast_size();
  const tcapint doutSize = dout.get_broadcast_size();
  if ((dinSize != inSize) || (dinSize != doutSize)) {
    throw std::invalid_argument(
        "In Weed::extremum_grad(din, in, dout), sizes do not match!");
  }
  if (in.storage->dtype != DType::REAL) {
    throw std::invalid_argument("In Weed::extremum_grad(din, in, dout), 'in' "
                                "dtype must be real-number!");
  }
  switch (din.storage->dtype) {
  case DType::COMPLEX:
    switch (dout.storage->dtype) {
    case DType::COMPLEX:
#if ENABLE_GPU
      DEVICE_SWITCH(cpu_grad_complex, gpu_grad_complex, din, in, dout, out);
#else
      cpu_grad_complex(din, in, dout, out);
#endif
      break;
    case DType::REAL:
    default:
#if ENABLE_GPU
      DEVICE_SWITCH(cpu_grad_mixed, gpu_grad_mixed, din, in, dout, out);
#else
      cpu_grad_complex(din, in, dout, out);
#endif
    }
    break;
  case DType::REAL:
  default:
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_grad_real, gpu_grad_real, din, in, dout, out);
#else
    cpu_grad_real(din, in, dout, out);
#endif
  }
}

RealExtremumKernel max_kernel = {
    cpu_max,        cpu_grad_real, cpu_grad_complex,
    cpu_grad_mixed,
#if ENABLE_GPU
    gpu_max,        gpu_grad_real, gpu_grad_complex,
    gpu_grad_mixed
#endif
};

RealExtremumKernel min_kernel = {
    cpu_min,        cpu_grad_real, cpu_grad_complex,
    cpu_grad_mixed,
#if ENABLE_GPU
    gpu_min,        gpu_grad_real, gpu_grad_complex,
    gpu_grad_mixed
#endif
};

void max(const Tensor &a, Tensor &out) { max_kernel.extremum(a, out); }
void max_grad(Tensor &din, const Tensor &in, const Tensor &dout,
              const Tensor &out) {
  max_kernel.extremum_grad(din, in, dout, out);
}

void min(const Tensor &a, Tensor &out) { min_kernel.extremum(a, out); }
void min_grad(Tensor &din, const Tensor &in, const Tensor &dout,
              const Tensor &out) {
  min_kernel.extremum_grad(din, in, dout, out);
}
} // namespace Weed
