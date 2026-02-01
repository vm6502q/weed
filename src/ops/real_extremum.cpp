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
#include "storage/all_storage.hpp"
#include "tensors/real_scalar.hpp"

#define CPU_STORAGE_INIT()                                                     \
  const tcapint O_a = a.offset;                                                \
  const tcapint I_a = a.stride[0U];                                            \
  GET_STORAGE(RealStorage, a, pa);                                             \
  const size_t n = a.get_broadcast_size()

#define CPU_GRAD_STORAGE_INIT_3(ft1, ft2, ft3)                                 \
  const tcapint O_d = din.offset;                                              \
  const tcapint I_d = din.stride[0U];                                          \
  const tcapint O_i = in.offset;                                               \
  const tcapint I_i = in.stride[0U];                                           \
  const tcapint O_o = dout.offset;                                             \
  const tcapint I_o = dout.stride[0U];                                         \
  GET_STORAGE(ft1, din, pdi);                                                  \
  GET_STORAGE(ft2, in, pi);                                                    \
  GET_STORAGE(ft3, dout, po);                                                  \
  const size_t n = din.get_broadcast_size()

#define GPU_GRAD(type1, type2, type3, api_call)                                \
  GPU_GRAD_ARGS();                                                             \
  std::shared_ptr<type1> a_storage =                                           \
      std::dynamic_pointer_cast<type1>(din.storage);                           \
  std::shared_ptr<type2> b_storage =                                           \
      std::dynamic_pointer_cast<type2>(in.storage);                            \
  std::shared_ptr<type3> c_storage =                                           \
      std::dynamic_pointer_cast<type3>(dout.storage);                          \
  const real1 m = static_cast<const RealScalar *>(&out)->get_item();           \
  const complex v = complex(m);                                                \
  a_storage->dev->RequestKernel(                                               \
      OCLAPI::api_call, args, din.get_size(),                                  \
      {a_storage->buffer, b_storage->buffer, c_storage->buffer}, 0U, &v)

#define DEVICE_SWITCH(cpu, gpu, din, in, dout, out)                            \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(din, in, dout, out);                                                   \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(din, in, dout, out);                                                   \
  }

#define GPU_GRAD_ARGS()                                                        \
  const tcapint args[10U] {                                                    \
    din.offset, din.stride[0U], in.offset, in.stride[0U], dout.offset,         \
        dout.stride[0U], 0U, 0U, 0U, 0U                                        \
  }

#define CPU_GRAD()                                                             \
  const real1 m = static_cast<const RealScalar *>(&out)->get_item();           \
  const auto fn = [&](const tcapint &i, const unsigned &cpu) {                 \
    if ((*pi)[O_i + i * I_i] == m) {                                           \
      pdi->add(O_d + i * I_d, (*po)[O_o + i * I_o]);                           \
    }                                                                          \
  };                                                                           \
  pfControl.par_for(0, n, fn)

#define CPU_MAX()                                                              \
  const unsigned cpuCount =                                                    \
      (unsigned)std::min(n, (size_t)pfControl.GetNumCores());                  \
  std::vector<real1> m(cpuCount, (*pa)[O_a]);                                  \
  pfControl.par_for(1U, n, [&](const tcapint &i, const unsigned &cpu) {        \
    const real1 v = (*pa)[O_a + i * I_a];                                      \
    if (v > m[cpu]) {                                                          \
      m[cpu] = v;                                                              \
    }                                                                          \
  });                                                                          \
  const real1 v = *std::max_element(m.begin(), m.end())

#define CPU_MIN()                                                              \
  const unsigned cpuCount =                                                    \
      (unsigned)std::min(n, (size_t)pfControl.GetNumCores());                  \
  std::vector<real1> m(cpuCount, (*pa)[O_a]);                                  \
  pfControl.par_for(1U, n, [&](const tcapint &i, const unsigned &cpu) {        \
    const real1 v = (*pa)[O_a + i * I_a];                                      \
    if (v < m[cpu]) {                                                          \
      m[cpu] = v;                                                              \
    }                                                                          \
  });                                                                          \
  const real1 v = *std::min_element(m.begin(), m.end())

#define GPU_HEADER()                                                           \
  if (!(a_storage.array)) {                                                    \
    a_storage.array = a_storage.Alloc(n);                                      \
  }                                                                            \
  real1 *gpa = a_storage.array.get();                                          \
  const bool isMapped =                                                        \
      a_storage.dev->LockSync(a_storage.buffer, sizeof(real1) * n, gpa)

#define GPU_WRITE(SetType)                                                     \
  o_storage.dev->SetType(v, o_storage.buffer, 0U);                             \
  if (isMapped) {                                                              \
    a_storage.dev->UnlockSync(a_storage.buffer, a_storage.array.get());        \
  } else {                                                                     \
    a_storage.array = nullptr;                                                 \
  }

#define GPU_CAST(storage1, storage2)                                           \
  storage1 a_storage = *static_cast<storage1 *>(a.storage.get());              \
  storage2 o_storage = *static_cast<storage2 *>(out.storage.get())

namespace Weed {
static void cpu_max(const Tensor &a, Tensor &out) {
  CPU_STORAGE_INIT();
  CPU_MAX();
  GET_STORAGE(RealStorage, out, po);
  po->write(0U, v);
}

static void cpu_min(const Tensor &a, Tensor &out) {
  CPU_STORAGE_INIT();
  CPU_MIN();
  GET_STORAGE(RealStorage, out, po);
  po->write(0U, v);
}

static void cpu_grad_real(Tensor &din, const Tensor &in, const Tensor &dout,
                          const Tensor &out) {
  CPU_GRAD_STORAGE_INIT_3(RealStorage, RealStorage, RealStorage);
  CPU_GRAD();
}
static void cpu_grad_complex(Tensor &din, const Tensor &in, const Tensor &dout,
                             const Tensor &out) {
  CPU_GRAD_STORAGE_INIT_3(ComplexStorage, RealStorage, ComplexStorage);
  CPU_GRAD();
}
static void cpu_grad_mixed(Tensor &din, const Tensor &in, const Tensor &dout,
                           const Tensor &out) {
  CPU_GRAD_STORAGE_INIT_3(ComplexStorage, RealStorage, RealStorage);
  CPU_GRAD();
}

#if ENABLE_GPU
static void gpu_max(const Tensor &a, Tensor &out) {
  CPU_STORAGE_INIT();
  GPU_CAST(GpuRealStorage, GpuRealStorage);
  GPU_HEADER();
  CPU_MAX();
  GPU_WRITE(SetReal);
}

static void gpu_min(const Tensor &a, Tensor &out) {
  CPU_STORAGE_INIT();
  GPU_CAST(GpuRealStorage, GpuRealStorage);
  GPU_HEADER();
  CPU_MIN();
  GPU_WRITE(SetReal);
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
  if (a.get_broadcast_size() != out.get_broadcast_size()) {
    throw std::invalid_argument("In RealExtremumKernel::extremum(a, out), out "
                                "size does not match input size!");
  }
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
