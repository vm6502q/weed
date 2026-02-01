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
struct RealExtremumKernel {
  void (*cpu)(const Tensor &, Tensor &);
  void (*cpu_grad_real)(Tensor &, const Tensor &, const Tensor &,
                        const Tensor &);
  void (*cpu_grad_complex)(Tensor &, const Tensor &, const Tensor &,
                           const Tensor &);
  void (*cpu_grad_mixed)(Tensor &, const Tensor &, const Tensor &,
                         const Tensor &);
#if ENABLE_GPU
  void (*gpu)(const Tensor &, Tensor &);
  void (*gpu_grad_real)(Tensor &, const Tensor &, const Tensor &,
                        const Tensor &);
  void (*gpu_grad_complex)(Tensor &, const Tensor &, const Tensor &,
                           const Tensor &);
  void (*gpu_grad_mixed)(Tensor &, const Tensor &, const Tensor &,
                         const Tensor &);
#endif
  void extremum(const Tensor &, Tensor &);
  void extremum_grad(Tensor &, const Tensor &, const Tensor &, const Tensor &);
};

extern RealExtremumKernel max_kernel;
extern RealExtremumKernel min_kernel;

/**
 * Maximum (real) extremum
 */
void max(const Tensor &a, Tensor &out);
/**
 * Maximum (real) extremum function gradient (all-matching)
 */
void max_grad(Tensor &din, const Tensor &in, const Tensor &dout,
              const Tensor &out);

/**
 * Minimum (real) extremum
 */
void min(const Tensor &a, Tensor &out);
/**
 * Minimum (real) extremum function gradient (all-matching)
 */
void min_grad(Tensor &din, const Tensor &in, const Tensor &dout,
              const Tensor &out);
} // namespace Weed
