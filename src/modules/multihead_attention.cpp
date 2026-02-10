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

#include "modules/multihead_attention.hpp"
#include "common/serializer.hpp"

namespace Weed {
TensorPtr MultiHeadAttention::forward(const TensorPtr x) {
  // x: (B, T, d_model)
  const auto &sh = x->shape;
  const symint B = sh[0];
  const symint T = sh[1];

  TensorPtr Q = W_q->forward(x);
  TensorPtr K = W_k->forward(x);
  TensorPtr V = W_v->forward(x);

  // (B, T, H, head_dim)
  Q = Tensor::reshape(Q, {B, T, num_heads, head_dim});
  K = Tensor::reshape(K, {B, T, num_heads, head_dim});
  V = Tensor::reshape(V, {B, T, num_heads, head_dim});

  // (B, H, T, head_dim)
  Q = Tensor::transpose(Q, 1, 2);
  K = Tensor::transpose(K, 1, 2);
  V = Tensor::transpose(V, 1, 2);

  // scores = Q K^T
  TensorPtr Kt = Tensor::transpose(K, -2, -1);
  TensorPtr scores = Q >> Kt;

  // scale
  scores = scores / real1(std::sqrt((real1)head_dim));

  // softmax over last axis
  TensorPtr weights = Tensor::softmax(scores, -1);

  // out = weights V
  TensorPtr out = weights >> V;

  // (B, T, H, head_dim)
  out = Tensor::transpose(out, 1, 2);

  // (B, T, d_model)
  out = Tensor::reshape(out, {B, T, d_model});

  // final projection
  return W_o->forward(out);
}
void MultiHeadAttention::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_symint(os, d_model);
  Serializer::write_symint(os, num_heads);
  Serializer::write_symint(os, head_dim);
  W_q->save(os);
  W_k->save(os);
  W_v->save(os);
  W_o->save(os);
}
} // namespace Weed
