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

#include "ops/triu_fill.hpp"
#include "common/parallel_for.hpp"
#include "ops/util.hpp"
#include "tensors/flat_tensors.hpp"

#define GPU(type, p, api_call)                                                 \
  GPU_ARGS();                                                                  \
  std::shared_ptr<type> a_storage =                                            \
      std::dynamic_pointer_cast<type>(a.storage);                              \
  const complex v = complex(p);                                                \
  a_storage->dev->RequestKernel(OCLAPI::api_call, args, a.shape[0],            \
                                {a_storage->buffer}, a.shape[1], &v)

#define DEVICE_SWITCH(cpu, gpu, a, b, d)                                       \
  switch (a.storage->device) {                                                 \
  case DeviceTag::GPU:                                                         \
    gpu(a, b, d);                                                              \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, b, d);                                                              \
  }

#define GPU_ARGS()                                                             \
  const tcapint args[10U] { a.shape[0], 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U }

namespace Weed {
template <typename T1, typename T2>
static void cpu_triu_fill(Tensor &a, const T1 &val, const tcapint diagonal) {
  if (a.shape.size() != 2U) {
    throw std::invalid_argument("triu_fill requires a 2D tensor!");
  }

  const tcapint t0 = a.shape[0]; // rows (col-major: fast index)
  const tcapint t1 = a.shape[1]; // cols

  GET_FLAT_TENSOR(T2, a, pa);

  pfControl.par_for(0, t0 * t1, [&](const tcapint &idx, const unsigned &) {
    const tcapint i = idx % t0; // row
    const tcapint j = idx / t0; // col
    if ((i + diagonal) <= j) {
      pa->write(idx, val);
    }
  });
}
static inline void cpu_real(Tensor &a, const complex &val,
                            const tcapint diagonal) {
  cpu_triu_fill<real1, RealTensor>(a, real(val), diagonal);
}
static inline void cpu_complex(Tensor &a, const complex &val,
                               const tcapint diagonal) {
  cpu_triu_fill<complex, ComplexTensor>(a, val, diagonal);
}
#if ENABLE_GPU
static void gpu_real(Tensor &a, const complex &val, const tcapint diagonal) {
  GPU(GpuRealStorage, val, OCL_API_TRIU_FILL_REAL);
}
static void gpu_complex(Tensor &a, const complex &val, const tcapint diagonal) {
  GPU(GpuComplexStorage, val, OCL_API_TRIU_FILL_COMPLEX);
}
#endif

void triu_fill(Tensor &a, const complex &val, const tcapint diagonal) {
  const bool isComplex = a.storage->dtype == DType::COMPLEX;
  if (isComplex) {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_complex, gpu_complex, a, val, diagonal);
#else
    cpu_complex(a, val, diagonal);
#endif
  } else {
#if ENABLE_GPU
    DEVICE_SWITCH(cpu_real, gpu_real, a, val, diagonal);
#else
    cpu_real(a, val, diagonal);
#endif
  }
}
} // namespace Weed
