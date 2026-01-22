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

namespace Weed {
struct CommutingKernel {
  CommutingOperation op;
  void (*cpu_real)(const Tensor &, const Tensor &, Tensor &);
  void (*cpu_complex)(const Tensor &, const Tensor &, Tensor &);
  void (*cpu_mixed)(const Tensor &, const Tensor &, Tensor &);
  void (*gpu_real)(const Tensor &, const Tensor &, Tensor &);
  void (*gpu_complex)(const Tensor &, const Tensor &, Tensor &);
  void (*gpu_mixed)(const Tensor &, const Tensor &, Tensor &);
  void (*commuting)(const Tensor &a, const Tensor &b, Tensor &out);
};

extern CommutingKernel commuting_kernel;
} // namespace Weed
