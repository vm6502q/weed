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
#include "storage.hpp"
#include "tensor.hpp"

#define _DEVICE_SWITCH(cpu, gpu, a, b, out)                                    \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, b, out);                                                            \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, b, out);                                                            \
  }

#define _DEVICE_SWITCH_INPLACE(cpu, gpu, a, out)                               \
  switch (out.storage->device) {                                               \
  case DeviceTag::GPU:                                                         \
    gpu(a, out);                                                               \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    cpu(a, out);                                                               \
  }

namespace Weed {
struct CommutingKernel {
  CommutingOperation op;
  void (*cpu_real)(const Tensor &, const Tensor &, Tensor &);
  void (*cpu_complex)(const Tensor &, const Tensor &, Tensor &);
  void (*cpu_mixed)(const Tensor &, const Tensor &, Tensor &);
  void (*gpu_real)(const Tensor &, const Tensor &, Tensor &);
  void (*gpu_complex)(const Tensor &, const Tensor &, Tensor &);
  void (*gpu_mixed)(const Tensor &, const Tensor &, Tensor &);

  void (*cpu_real_inplace)(Tensor &, const Tensor &);
  void (*cpu_complex_inplace)(Tensor &, const Tensor &);
  void (*cpu_mixed_inplace)(Tensor &, const Tensor &);
  void (*gpu_real_inplace)(Tensor &, const Tensor &);
  void (*gpu_complex_inplace)(Tensor &, const Tensor &);
  void (*gpu_mixed_inplace)(Tensor &, const Tensor &);

  void commuting(const Tensor &a, const Tensor &b, Tensor &out) {
    const bool isAComplex = a.storage->dtype == DType::COMPLEX;
    const bool isBComplex = b.storage->dtype == DType::COMPLEX;
    const bool isOutComplex = out.storage->dtype == DType::COMPLEX;
    if (!isOutComplex && (isAComplex || isBComplex)) {
      throw std::invalid_argument(
          "Cannot combine complex tensors into real1 tensor!");
    }
    if (isOutComplex && (!isAComplex && !isBComplex)) {
      throw std::invalid_argument("Output tensor dtype mismatch!");
    }
    if (isAComplex && isBComplex) {
      _DEVICE_SWITCH(cpu_complex, gpu_complex, a, b, out);
    } else if (isAComplex) {
      _DEVICE_SWITCH(cpu_mixed, gpu_mixed, a, b, out);
    } else if (isBComplex) {
      _DEVICE_SWITCH(cpu_mixed, gpu_mixed, b, a, out);
    } else {
      _DEVICE_SWITCH(cpu_real, gpu_real, a, b, out);
    }
  }

  void commuting_inplace(Tensor &a, const Tensor &b) {
    const bool isAComplex = a.storage->dtype == DType::COMPLEX;
    const bool isBComplex = b.storage->dtype == DType::COMPLEX;
    if (isAComplex != isBComplex) {
      throw std::runtime_error("Output tensor dtype mismatch!");
    }
    if (isAComplex) {
      _DEVICE_SWITCH_INPLACE(cpu_complex_inplace, gpu_complex_inplace, a, b);
    } else {
      _DEVICE_SWITCH_INPLACE(cpu_real_inplace, gpu_real_inplace, a, b);
    }
  }
};

extern CommutingKernel commuting_kernel;
} // namespace Weed
