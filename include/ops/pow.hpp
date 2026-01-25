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

#include "tensors/real_scalar.hpp"

namespace Weed {
struct PowKernel {
  void cpu_real(const Tensor &, const RealScalar &, Tensor &);
  void cpu_complex(const Tensor &, const RealScalar &, Tensor &);
  void gpu_real(const Tensor &, const RealScalar &, Tensor &);
  void gpu_complex(const Tensor &, const RealScalar &, Tensor &);
  void pow(const Tensor &, const RealScalar &, Tensor &);
};

extern PowKernel pow_kernel;

/**
 * Element-wise power
 */
void pow(const Tensor &a, const RealScalar &p, Tensor &out);
} // namespace Weed
