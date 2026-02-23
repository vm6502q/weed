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
/**
 * Binary cross-entropy loss (with logistic simplification)
 */
inline TensorPtr bci_with_logits_loss(TensorPtr logits, TensorPtr y_true) {
  return Tensor::relu(logits) - logits * y_true +
         Tensor::log(ONE_R1 + Tensor::exp(-ONE_R1 * Tensor::abs(logits)));
}
} // namespace Weed
