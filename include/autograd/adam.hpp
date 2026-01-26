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

#include "tensors/parameter.hpp"

#include <unordered_map>

namespace Weed {
/**
 * Model moments of Adam optimizer
 */
struct AdamState {
  TensorPtr m; // first moment
  TensorPtr v; // second moment
};

/**
 * Adam optimizer
 */
struct Adam {
  real1 lr;
  real1 beta1;
  real1 beta2;
  real1 eps;
  uint64_t t;

  std::unordered_map<ParameterPtr, AdamState> state;

  Adam(real1 l, real1 b1, real1 b2, real1 e)
      : lr(l), beta1(b1), beta2(b2), eps(e), t(0U) {}

  /**
   * Register a parameter with this optimizer
   */
  void register_parameter(ParameterPtr p) {
    AdamState s;
    s.m = Tensor::allocate_like(p, p->storage->dtype, false);
    s.v = Tensor::allocate_like(p, p->storage->dtype, false);

    s.m->storage->FillZeros();
    s.v->storage->FillZeros();

    state[p] = s;
  }

  /**
   * Register a vector of parameters with this optimizer
   */
  void register_parameters(const std::vector<ParameterPtr> &pv) {
    for (const ParameterPtr &p : pv) {
      register_parameter(p);
    }
  }
};

void adam_step(Adam &opt, const std::vector<ParameterPtr> &params) {
  opt.t += 1;

  const real1 bias_correction1 = 1.0 - std::pow(opt.beta1, opt.t);
  const real1 bias_correction2 = 1.0 - std::pow(opt.beta2, opt.t);

  for (auto &p : params) {
    if (!p->grad)
      continue;

    AdamState &s = opt.state[p];
    // Tensor& m = *(s.m.get());
    // Tensor& v = *(s.v.get());
    TensorPtr g = p->grad;

    // m = beta1 * m + (1 - beta1) * g
    s.m = opt.beta1 * s.m + (1 - opt.beta1) * g;

    // v = beta2 * v + (1 - beta2) * g * g
    s.v = opt.beta2 * s.v + (1 - opt.beta2) * g * g;
    // i.e., v += (1-beta2) * (g ⊙ g)

    // Compute bias-corrected step
    // tmp = m / bias_correction1
    // tmp2 = v / bias_correction2
    // p -= lr * tmp / (sqrt(tmp2) + eps)
    TensorPtr tmp =
        opt.lr * s.m /
        (bias_correction1 * (((s.v / bias_correction2) ^ 0.5) + opt.eps));

    Weed::sub_in_place(*(p.get()), *(tmp.get()));
  }
}
} // namespace Weed
