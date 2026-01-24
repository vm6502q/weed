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
struct DivKernel {
  void cpu_real(const Tensor &, const Tensor &, Tensor &);
  void cpu_complex(const Tensor &, const Tensor &, Tensor &);
  void cpu_mixed_c_left(const Tensor &, const Tensor &, Tensor &);
  void cpu_mixed_c_right(const Tensor &, const Tensor &, Tensor &);
#if ENABLE_GPU
  void gpu_real(const Tensor &, const Tensor &, Tensor &);
  void gpu_complex(const Tensor &, const Tensor &, Tensor &);
  void gpu_mixed_c_left(const Tensor &, const Tensor &, Tensor &);
  void gpu_mixed_c_right(const Tensor &, const Tensor &, Tensor &);
#endif
  void div(const Tensor &, const Tensor &, Tensor &);
};

extern DivKernel div_kernel;

/**
 * Element-wise division
 */
void div(const Tensor &a, const Tensor &b, Tensor &out);
} // namespace Weed
