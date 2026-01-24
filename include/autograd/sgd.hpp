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
#include "tensors/real_scalar.hpp"

namespace Weed {
inline void sgd_step(const std::vector<TensorPtr> &params, real1 lr) {
  if (params.empty()) {
    return;
  }

  const DeviceTag dtag = params[0]->storage->device;
  TensorPtr alpha = std::make_shared<RealScalar>(lr, false, dtag);

  for (auto &p : params) {
    if (!p->grad) {
      continue;
    }

    const DType dt = Tensor::get_dtype_by_presidence(p, p->grad);
    p->upcast(dt);
    alpha->match_shape(p->grad);

    TensorPtr tmp = Tensor::allocate_like(p->grad, dt, false);
    Weed::mul(*(alpha.get()), *(p->grad.get()), *(tmp.get()));
    Weed::add_in_place(*(p.get()), *(tmp.get()));
  }
}
} // namespace Weed
