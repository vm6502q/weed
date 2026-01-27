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
struct ReduceKernel {
  void cpu_real(const size_t &, const Tensor &, Tensor &);
  void cpu_complex(const size_t &, const Tensor &, Tensor &);
#if ENABLE_GPU
  void gpu_real(const size_t &, const Tensor &, Tensor &);
  void gpu_complex(const size_t &, const Tensor &, Tensor &);
#endif
  void reduce(const size_t &, const Tensor &, Tensor &);
};

extern ReduceKernel reduce_kernel;

/**
 * Sum over a tensor index
 */
void reduce(const size_t &index, const Tensor &a, Tensor &out);
void reduce_broadcast(const std::vector<tcapint> stride, const Tensor &a,
                      Tensor &out);
} // namespace Weed
