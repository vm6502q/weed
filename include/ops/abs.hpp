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
struct AbsKernel {
  void cpu_real(const Tensor &, Tensor &);
  void cpu_complex(const Tensor &, Tensor &);
  void abs(const Tensor &, Tensor &);
  void cpu_real_grad_real(Tensor &, const Tensor &, const Tensor &);
  void cpu_real_grad_complex(Tensor &, const Tensor &, const Tensor &);
  void cpu_complex_grad_real(Tensor &, const Tensor &, const Tensor &);
  void cpu_complex_grad_complex(Tensor &, const Tensor &, const Tensor &);
#if ENABLE_GPU
  void gpu_real(const Tensor &, Tensor &);
  void gpu_complex(const Tensor &, Tensor &);
  void gpu_real_grad_real(Tensor &, const Tensor &, const Tensor &);
  void gpu_real_grad_complex(Tensor &, const Tensor &, const Tensor &);
  void gpu_complex_grad_real(Tensor &, const Tensor &, const Tensor &);
  void gpu_complex_grad_complex(Tensor &, const Tensor &, const Tensor &);
#endif
  void abs_grad(Tensor &, const Tensor &, const Tensor &);
};

extern AbsKernel abs_kernel;

/**
 * Absolute value
 */
void abs(const Tensor &a, Tensor &out);
/**
 * Absolute value gradient
 */
void abs_grad(Tensor &din, const Tensor &in, const Tensor &dout);
} // namespace Weed
