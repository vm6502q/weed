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

#pragma once

#include "commuting_operation.hpp"
#include "tensor.hpp"
#include "storage.hpp"

#define _DEVICE_SWITCH(cpu, gpu, a, b)                                         \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, b, out);                                                            \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, b, out);                                                            \
  }

namespace Weed {
struct CommutingKernel {
  CommutingOperation op;
  void (*cpu_real)(const Tensor &, const Tensor &, Tensor &);
  void (*cpu_complex)(const Tensor &, const Tensor &, Tensor &);
  void (*cpu_mixed)(const Tensor &, const Tensor &, Tensor &);
  void (*cpu_promote)(const Tensor &, const Tensor &, Tensor &);
  void (*gpu_real)(const Tensor &, const Tensor &, Tensor &);
  void (*gpu_complex)(const Tensor &, const Tensor &, Tensor &);
  void (*gpu_mixed)(const Tensor &, const Tensor &, Tensor &);
  void (*gpu_promote)(const Tensor &, const Tensor &, Tensor &);

  void commuting(const Tensor &a, const Tensor &b, Tensor &out) {
    const bool isAComplex = a.storage->dtype == DType::COMPLEX;
    const bool isBComplex = b.storage->dtype == DType::COMPLEX;
    const bool isOutComplex = out.storage->dtype == DType::COMPLEX;
    if (!isOutComplex && (isAComplex || isBComplex)) {
      throw std::runtime_error(
          "Cannot combine complex tensors into real1 tensor!");
    }
    if (isAComplex && isBComplex) {
      _DEVICE_SWITCH(cpu_complex, gpu_complex, a, b);
    } else if (isAComplex) {
      _DEVICE_SWITCH(cpu_mixed, gpu_mixed, a, b);
    } else if (isBComplex) {
      _DEVICE_SWITCH(cpu_mixed, gpu_mixed, b, a);
    } else if (isOutComplex) {
      _DEVICE_SWITCH(cpu_promote, gpu_promote, a, b);
    } else {
      _DEVICE_SWITCH(cpu_real, gpu_real, a, b);
    }
  }
};

extern CommutingKernel commuting_kernel;
} // namespace Weed
