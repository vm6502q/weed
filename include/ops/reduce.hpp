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
  void cpu_real(const tcapint &, const Tensor &, Tensor &);
  void cpu_complex(const tcapint &, const Tensor &, Tensor &);
  void cpu_grad_real(const tcapint &, Tensor &, const Tensor &, const Tensor &);
  void cpu_grad_complex(const tcapint &, Tensor &, const Tensor &,
                        const Tensor &);
  void cpu_grad_mixed(const tcapint &, Tensor &, const Tensor &,
                      const Tensor &);
#if ENABLE_GPU
  void gpu_real(const tcapint &, const Tensor &, Tensor &);
  void gpu_complex(const tcapint &, const Tensor &, Tensor &);
  void gpu_grad_real(const tcapint &, Tensor &, const Tensor &, const Tensor &);
  void gpu_grad_complex(const tcapint &, Tensor &, const Tensor &,
                        const Tensor &);
  void gpu_grad_mixed(const tcapint &, Tensor &, const Tensor &,
                      const Tensor &);
#endif
  void reduce(const tcapint &, const Tensor &, Tensor &);
  void reduce_grad(const tcapint &, Tensor &, const Tensor &, const Tensor &);
};

extern ReduceKernel reduce_kernel;

/**
 * Sum over a tensor index
 */
void reduce(const tcapint &index, const Tensor &a, Tensor &out);
void reduce_grad(const tcapint &index, Tensor &din, const Tensor &a,
                 const Tensor &dout);
void reduce_broadcast(const std::vector<tcapint> stride, const Tensor &a,
                      Tensor &out);
} // namespace Weed
