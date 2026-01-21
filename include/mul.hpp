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

#include "commuting.hpp"

namespace Weed {
struct MulKernel : CommutingKernel {
  MulKernel() { op = CommutingOperation::MUL; }
};

MulKernel mul_kernel;

void mul(const Tensor &a, const Tensor &b, Tensor &out) {
  mul_kernel.commuting(a, b, out);
}

void mul_inplace(Tensor &a, const Tensor& b) {
  mul_kernel.commuting_inplace(a, b);
}
} // namespace Weed
