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

#include "ops/embedding.hpp"
#include "tensors/tensor.hpp"

namespace Weed {
/**
 * Cross-entropy loss (-mean(logsoftmax(logits)[range(T), targets]))
 */
static TensorPtr cross_entropy_loss(TensorPtr logits, SymbolTensorPtr targets) {
  // logits: [1, seq_len, vocab_size]
  const symint T = logits->shape[1];
  const symint V = logits->shape[2];

  TensorPtr lsm = Tensor::logsoftmax(logits, -1);
  lsm = Tensor::reshape(lsm, {T, V});

  TensorPtr oh = Tensor::one_hot(targets, (tcapint)V); // sparse [T, vocab_size]
  TensorPtr selected = lsm * oh;                 // [T, vocab_size], sparse
  TensorPtr gathered = Tensor::sum(selected, 1); // [T, 1]

  return Tensor::mean(gathered) * real1(-1.0f);
}
} // namespace Weed
