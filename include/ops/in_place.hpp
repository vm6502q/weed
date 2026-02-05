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
struct InPlaceKernel {
  void (*cpu_real)(Tensor &, const Tensor &);
  void (*cpu_complex)(Tensor &, const Tensor &);
  void (*cpu_mixed)(Tensor &, const Tensor &);
#if ENABLE_GPU
  void (*gpu_real)(Tensor &, const Tensor &);
  void (*gpu_complex)(Tensor &, const Tensor &);
  void (*gpu_mixed)(Tensor &, const Tensor &);
#endif
  void in_place(Tensor &a, const Tensor &b);
};

/**
 * Element-wise add-in-place
 */
void add_in_place(Tensor &a, const Tensor &b);
/**
 * Element-wise subtract-in-place
 */
void sub_in_place(Tensor &a, const Tensor &b);
} // namespace Weed
