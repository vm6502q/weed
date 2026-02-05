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
struct RealUnaryKernel {
  void (*cpu)(const Tensor &, Tensor &);
  void (*cpu_grad_real)(Tensor &, const Tensor &, const Tensor &);
  void (*cpu_grad_complex)(Tensor &, const Tensor &, const Tensor &);
  void (*cpu_grad_mixed)(Tensor &, const Tensor &, const Tensor &);
#if ENABLE_GPU
  void (*gpu)(const Tensor &, Tensor &);
  void (*gpu_grad_real)(Tensor &, const Tensor &, const Tensor &);
  void (*gpu_grad_complex)(Tensor &, const Tensor &, const Tensor &);
  void (*gpu_grad_mixed)(Tensor &, const Tensor &, const Tensor &);
#endif
  void unary(const Tensor &, Tensor &);
  void unary_grad(Tensor &, const Tensor &, const Tensor &);
};

/**
 * Rectified-linear activation function
 */
void relu(const Tensor &a, Tensor &out);
/**
 * Rectified-linear activation function gradient
 */
void relu_grad(Tensor &din, const Tensor &in, const Tensor &dout);

/**
 * Sigmoid activation function
 */
void sigmoid(const Tensor &a, Tensor &out);
/**
 * Sigmoid activation function gradient
 */
void sigmoid_grad(Tensor &din, const Tensor &in, const Tensor &dout);

/**
 * Hyberbolic tangent activation function
 */
void tanh(const Tensor &a, Tensor &out);
/**
 * Hyberbolic tangent activation function gradient
 */
void tanh_grad(Tensor &din, const Tensor &in, const Tensor &dout);
} // namespace Weed
