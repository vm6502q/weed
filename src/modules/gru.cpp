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

namespace Weed {
TensorPtr GRU::forward(const TensorPtr x) {
  const TensorPtr &prev = state.back();
  if (prev->shape.size() == 1U) {
    prev->shape.insert(prev->shape.begin(), x->shape[0U]);
    prev->stride.insert(prev->stride.begin(), 0U);
    prev->materialize_broadcast();
  }

  // z = W_x(x) + W_h(h)
  TensorPtr z = W_x.forward(x) + W_h.forward(prev);

  // Split into 3 chunks
  const std::vector<TensorPtr> zc = z->chunk(3, /*axis=*/-1);

  TensorPtr z_t = Tensor::sigmoid(zc[0U]);
  TensorPtr r_t = Tensor::sigmoid(zc[1U]);

  // Candidate
  TensorPtr h_tilde = Tensor::tanh(zc[2U] + W_h.forward(r_t * prev));

  // Final hidden state
  TensorPtr h = (Tensor::ones_like(z_t->shape) - z_t) * prev + z_t * h_tilde;

  state.push_back(h);

  return h;
}
} // namespace Weed
