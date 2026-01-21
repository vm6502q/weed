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
struct ReluKernel {
  void (*cpu_real)(const Tensor &, Tensor &);
  void (*gpu_real)(const Tensor &, Tensor &);
  void (*relu)(const Tensor &, Tensor &);
  void (*cpu_real_grad)(StoragePtr, const Tensor &, const StoragePtr);
  void (*gpu_real_grad)(StoragePtr, const Tensor &, const StoragePtr);
  void (*relu_grad)(StoragePtr, const Tensor &, const StoragePtr);
};

ReluKernel relu_kernel;

void relu(const Tensor &a, Tensor &out) { relu_kernel.relu(a, out); }
void relu_grad(StoragePtr din, const Tensor& in, const StoragePtr dout) { relu_kernel.relu_grad(din, in, dout); }
} // namespace Weed
