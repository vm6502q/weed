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
struct CommutingKernel {
  void (*cpu_real)(const Tensor &, const Tensor &, Tensor &);
  void (*cpu_complex)(const Tensor &, const Tensor &, Tensor &);
  void (*cpu_mixed)(const Tensor &, const Tensor &, Tensor &);
#if ENABLE_GPU
  void (*gpu_real)(const Tensor &, const Tensor &, Tensor &);
  void (*gpu_complex)(const Tensor &, const Tensor &, Tensor &);
  void (*gpu_mixed)(const Tensor &, const Tensor &, Tensor &);
#endif
  void commuting(const Tensor &a, const Tensor &b, Tensor &out);
};

/**
 * Element-wise addition
 */
void add(const Tensor &a, const Tensor &b, Tensor &out);
/**
 * Element-wise multiplication
 */
void mul(const Tensor &a, const Tensor &b, Tensor &out);
} // namespace Weed
