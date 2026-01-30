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

#define RSCALAR(v, o)                                                          \
  std::make_shared<RealScalar>(v, false, o->storage->device,                   \
                               o->storage->get_device_id())

namespace Weed {
/**
 * Binary cross-entropy loss
 */
inline TensorPtr bci_loss(TensorPtr y_pred, TensorPtr y_true) {
  WEED_CONST real1 eps(100 * FP_NORM_EPSILON);
  y_pred = Tensor::clamp(y_pred, eps, ONE_R1 - eps);
  TensorPtr unit = RSCALAR(ONE_R1, y_pred);

  return Tensor::mean((y_true - unit) * Tensor::log(unit - y_pred) -
                      y_true * Tensor::log(y_pred));
}
} // namespace Weed
