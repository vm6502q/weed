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

#include "ops/commuting.hpp"
#include "ops/in_place.hpp"
#include "tensors/parameter.hpp"
#include "tensors/real_scalar.hpp"

namespace Weed {
/**
 * Stochastic gradient descent (SGD) optimization step
 */
inline void sgd_step(const std::vector<ParameterPtr> &params, real1 lr) {
  if (params.empty()) {
    return;
  }

  for (auto &p : params) {
    // A parameter that was broadcast in the forward pass (like a bias) only
    // stores one element per broadcast index. Without this, the update below
    // would be applied to that element once for every broadcast position.
    p->reset_broadcast();
    TensorPtr pg = p->grad;
    TensorPtr tmp = lr * pg;
    tmp->match_shape(p);
    const DeviceTag dtag = Tensor::get_dtag_by_presidence({p, tmp});
    tmp = tmp->cast(dtag);
    p->cast_in_place(dtag);
    Weed::sub_in_place(*(p.get()), *(tmp.get()));
  }
}
} // namespace Weed
