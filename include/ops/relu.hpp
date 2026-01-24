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

#include "tensors/tensor.hpp"

namespace Weed {
struct ReluKernel {
  void cpu(const Tensor &, Tensor &);
  void relu(const Tensor &, Tensor &);
  void cpu_grad_real(Tensor &, const Tensor &, const Tensor &);
  void cpu_grad_complex(Tensor &, const Tensor &, const Tensor &);
#if ENABLE_GPU
  void gpu(const Tensor &, Tensor &);
  void gpu_grad_real(Tensor &, const Tensor &, const Tensor &);
  void gpu_grad_complex(Tensor &, const Tensor &, const Tensor &);
#endif
  void relu_grad(Tensor &, const Tensor &, const Tensor &);
};

extern ReluKernel relu_kernel;

/**
 * Rectified-linear activation function
 */
void relu(const Tensor &a, Tensor &out);
/**
 * Rectified-linear activation function gradient
 */
void relu_grad(Tensor &din, const Tensor &in, const Tensor &dout);
} // namespace Weed
