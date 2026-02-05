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
struct PowKernel {
  void (*cpu_real)(const Tensor &, const real1 &, Tensor &);
  void (*cpu_complex)(const Tensor &, const real1 &, Tensor &);
#if ENABLE_GPU
  void (*gpu_real)(const Tensor &, const real1 &, Tensor &);
  void (*gpu_complex)(const Tensor &, const real1 &, Tensor &);
#endif
  void pow(const Tensor &, const real1 &, Tensor &);
};

/**
 * Element-wise power
 */
void pow(const Tensor &a, const real1 &p, Tensor &out);
/**
 * Element-wise exponential
 */
void exp(const Tensor &a, const real1 &p, Tensor &out);
/**
 * Element-wise logarithm
 */
void log(const Tensor &a, const real1 &b, Tensor &out);
} // namespace Weed
