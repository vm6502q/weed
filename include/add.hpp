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

#include "tensor.hpp"

namespace Weed {
struct AddKernel {
  void (*cpu_real)(const Tensor &, const Tensor &, Tensor &);
  void (*cpu_complex)(const Tensor &, const Tensor &, Tensor &);
  void (*opencl_real)(const Tensor &, const Tensor &, Tensor &);
  void (*opencl_complex)(const Tensor &, const Tensor &, Tensor &);
};

extern AddKernel add_kernel;

void add(const Tensor &a, const Tensor &b, Tensor &out) {
  switch (out.storage->dtype) {
  case DType::COMPLEX:
    switch (out.storage->device) {
    case DeviceTag::OpenCL:
      add_kernel.opencl_complex(a, b, out);
      break;
    case DeviceTag::CPU:
    default:
      add_kernel.cpu_complex(a, b, out);
      break;
    }
    break;
  case DType::REAL:
  default:
    switch (out.storage->device) {
    case DeviceTag::OpenCL:
      add_kernel.opencl_real(a, b, out);
      break;
    case DeviceTag::CPU:
    default:
      add_kernel.cpu_real(a, b, out);
      break;
    }
  }
}
} // namespace Weed
