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

#include "relu.hpp"
#include "common/parallel_for.hpp"
#include "cpu_real_storage.hpp"
#include "gpu_real_storage.hpp"

#define CAST_STORAGE(out, in, type, ptr)                                       \
  type *out = static_cast<ptr *>(in.storage.get())->data.get() + in.offset

namespace Weed {
ParallelFor pfControl = ParallelFor();

struct relu_kernel : ReluKernel {
  void cpu_real(const Tensor &a, Tensor &out) {
    CAST_STORAGE(pa, a, real1, CpuRealStorage);
    CAST_STORAGE(po, out, real1, CpuRealStorage);
    size_t n = out.storage->size;
    pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
      po[i] = std::max(pa[i], ZERO_R1);
    });
  }
  void gpu_real(const Tensor &a, Tensor &out) {
    const vecCapIntGpu args[3U]{a.offset, out.offset, 0U};
    GpuRealStoragePtr a_storage =
        std::dynamic_pointer_cast<GpuRealStorage>(a.storage);
    GpuRealStoragePtr o_storage =
        std::dynamic_pointer_cast<GpuRealStorage>(out.storage);
    a_storage->gpu->RequestKernel(OCLAPI::OCL_API_RELU, args, a.get_size(),
                                  {a_storage->buffer, o_storage->buffer});
  }
  void relu(const Tensor &a, Tensor &out) {
    if ((a.storage->dtype == DType::COMPLEX) or
        (out.storage->dtype == DType::COMPLEX)) {
      throw std::invalid_argument("Cannot apply ReLU on complex tensors!");
    }
    switch (out.storage->device) {
    case DeviceTag::GPU:
      gpu_real(a, out);
      break;
    case DeviceTag::CPU:
    default:
      cpu_real(a, out);
    }
  }

  void cpu_real_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
    CAST_STORAGE(pdi, din, real1, CpuRealStorage);
    CAST_STORAGE(pi, in, real1, CpuRealStorage);
    CAST_STORAGE(po, dout, real1, CpuRealStorage);
    size_t n = dout.storage->size;
    pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
      pdi[i] = (pi[i] > 0) ? po[i] : ZERO_R1;
    });
  }
  void gpu_real_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
    const vecCapIntGpu args[3U]{din.offset, in.offset, dout.offset};
    GpuRealStoragePtr a_storage =
        std::dynamic_pointer_cast<GpuRealStorage>(din.storage);
    GpuRealStoragePtr b_storage =
        std::dynamic_pointer_cast<GpuRealStorage>(in.storage);
    GpuRealStoragePtr c_storage =
        std::dynamic_pointer_cast<GpuRealStorage>(dout.storage);
    a_storage->gpu->RequestKernel(
        OCLAPI::OCL_API_RELU_GRAD, args, din.get_size(),
        {a_storage->buffer, b_storage->buffer, c_storage->buffer});
  }

  void relu_grad(Tensor &din, const Tensor &in, const Tensor &dout) {
    switch (din.storage->device) {
    case DeviceTag::GPU:
      gpu_real_grad(din, in, dout);
      break;
    case DeviceTag::CPU:
    default:
      cpu_real_grad(din, in, dout);
    }
  }
};
} // namespace Weed
