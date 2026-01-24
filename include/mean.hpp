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

#include "tensor.hpp"

namespace Weed {
struct MeanKernel {
  void cpu_real(const Tensor &, Tensor &);
  void cpu_complex(const Tensor &, Tensor &);
#if ENABLE_GPU
  void gpu_real(const Tensor &, Tensor &);
  void gpu_complex(const Tensor &, Tensor &);
#endif
  void mean(const Tensor &, Tensor &);
};

extern MeanKernel mean_kernel;

void mean(const Tensor &a, Tensor &out);
} // namespace Weed
