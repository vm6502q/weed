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
struct ReluKernel {
  void (*cpu_real)(const Tensor &, Tensor &);
  void (*gpu_real)(const Tensor &, Tensor &);
  void (*relu)(const Tensor &, Tensor &);
};

extern ReluKernel relu_kernel;

void relu(const Tensor &a, Tensor &out) { relu_kernel.relu(a, out); }
} // namespace Weed
