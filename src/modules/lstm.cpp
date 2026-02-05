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

#include "modules/lstm.hpp"

namespace Weed {
TensorPtr LSTM::forward(const TensorPtr x) {
  const LSTMState &prev = state.back();

  // z = W_x(x) + W_h(h_{t-1})
  TensorPtr z = W_x.forward(x) + W_h.forward(prev.h);

  // Split z into 4 chunks
  const std::vector<TensorPtr> zc = z->chunk(4, -1);

  TensorPtr f = Tensor::sigmoid(zc[0U]);
  TensorPtr i = Tensor::sigmoid(zc[1U]);
  TensorPtr g = Tensor::tanh(zc[2U]);
  TensorPtr o = Tensor::sigmoid(zc[3U]);

  TensorPtr c = f * prev.c + i * g;
  TensorPtr h = o * Tensor::tanh(c);

  state.push_back({h, c});

  return h;
}
} // namespace Weed
