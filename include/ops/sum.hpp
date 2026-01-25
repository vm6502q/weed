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
struct SumKernel {
  void (*cpu_real)(const Tensor &, Tensor &);
  void (*cpu_complex)(const Tensor &, Tensor &);
#if ENABLE_GPU
  void (*gpu_real)(const Tensor &, Tensor &);
  void (*gpu_complex)(const Tensor &, Tensor &);
#endif
  void sum(const Tensor &, Tensor &);
};

extern SumKernel sum_kernel;
extern SumKernel mean_kernel;

/**
 * Sum of all elements
 */
void sum(const Tensor &a, Tensor &out);

/**
 * Average of all elements
 */
void mean(const Tensor &a, Tensor &out);
} // namespace Weed
