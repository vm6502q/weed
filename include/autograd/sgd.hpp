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

  TensorPtr alpha =
      std::make_shared<RealScalar>(lr, false, params[0U]->storage->device,
                                   params[0U]->storage->get_device_id());

  for (auto &p : params) {
    p->upcast(Tensor::get_dtype_by_presidence({p, p->grad}));
    TensorPtr tmp = alpha * p->grad;
    Weed::sub_in_place(*(p.get()), *(tmp.get()));
  }
}
} // namespace Weed
