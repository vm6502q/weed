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

#include "modules/gru.hpp"
#include "common/serializer.hpp"

namespace Weed {
TensorPtr GRU::forward(const TensorPtr x) {
  if (state->shape.size() == 1U) {
    state->shape.insert(state->shape.begin(), x->shape[0U]);
    state->stride.insert(state->stride.begin(), 0U);
    state->materialize_broadcast();
  }

  // z = W_x(x) + W_h(h)
  TensorPtr z = W_x->forward(x) + W_h->forward(state);

  // Split into 3 chunks
  const std::vector<TensorPtr> zc = Tensor::chunk(z, 3, /*axis=*/-1);

  TensorPtr z_t = Tensor::sigmoid(zc[0U]);
  TensorPtr r_t = Tensor::sigmoid(zc[1U]);

  // Candidate
  TensorPtr h_tilde = Tensor::tanh(zc[2U] + W_h->forward(r_t * state));

  // Final hidden state
  TensorPtr h = (Tensor::ones_like(z_t->shape) - z_t) * state + z_t * h_tilde;

  return h;
}
void GRU::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, input_dim);
  Serializer::write_tcapint(os, hidden_dim);
  W_x->save(os);
  W_h->save(os);
}
} // namespace Weed
