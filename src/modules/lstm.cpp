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
#include "common/serializer.hpp"

namespace Weed {
TensorPtr LSTM::forward(const TensorPtr x) {
  if (state.h->shape.size() == 1U) {
    state.h->shape.insert(state.h->shape.begin(), x->shape[0U]);
    state.h->stride.insert(state.h->stride.begin(), 0U);
    state.h->materialize_broadcast();
  }
  if (state.c->shape.size() == 1U) {
    state.c->shape.insert(state.c->shape.begin(), x->shape[0U]);
    state.c->stride.insert(state.c->stride.begin(), 0U);
    state.c->materialize_broadcast();
  }

  // z = W_x(x) + W_h(h_{t-1})
  TensorPtr z = W_x->forward(x) + W_h->forward(state.h);

  // Split z into 4 chunks
  const std::vector<TensorPtr> zc = Tensor::chunk(z, 4, -1);

  TensorPtr f = Tensor::sigmoid(zc[0U]);
  TensorPtr i = Tensor::sigmoid(zc[1U]);
  TensorPtr g = Tensor::tanh(zc[2U]);
  TensorPtr o = Tensor::sigmoid(zc[3U]);

  TensorPtr c = f * state.c + i * g;
  TensorPtr h = o * Tensor::tanh(c);

  state.h = h;
  state.c = c;

  return h;
}
void LSTM::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, input_dim);
  Serializer::write_tcapint(os, hidden_dim);
  W_x->save(os);
  W_h->save(os);
}
} // namespace Weed
