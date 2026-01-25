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
struct ClampKernel {
  void cpu(const Tensor &, const real1 &, const real1 &, Tensor &);
  void cpu_grad_real(const Tensor &, const Tensor &, const real1 &,
                     const real1 &, Tensor &);
  void cpu_grad_complex(const Tensor &, const Tensor &, const real1 &,
                        const real1 &, Tensor &);
#if ENABLE_GPU
  void gpu(const Tensor &, const real1 &, const real1 &, Tensor &);
  void gpu_grad_real(const Tensor &, const Tensor &, const real1 &,
                     const real1 &, Tensor &);
  void gpu_grad_complex(const Tensor &, const Tensor &, const real1 &,
                        const real1 &, Tensor &);
#endif
  void clamp(const Tensor &, const real1 &, const real1 &, Tensor &);
  void clamp_grad(const Tensor &, const Tensor &, const real1 &, const real1 &,
                  Tensor &);
};

extern ClampKernel clamp_kernel;

/**
 * Element-wise clamp
 */
void clamp(const Tensor &a, const real1 &l, const real1 &h, Tensor &out);
/**
 * Element-wise clamp gradient
 */
void clamp_grad(const Tensor &dy, const Tensor &x, const real1 &l,
                const real1 &h, Tensor &dx);
} // namespace Weed
