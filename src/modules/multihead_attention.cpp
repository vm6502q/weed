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
#include "ops/in_place.hpp"
#include "ops/triu_fill.hpp"

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
  Q = std::dynamic_pointer_cast<Tensor>(
      Tensor::reshape(Q, {B, T, num_heads, head_dim}));
  K = std::dynamic_pointer_cast<Tensor>(
      Tensor::reshape(K, {B, T, num_heads, head_dim}));
  V = std::dynamic_pointer_cast<Tensor>(
      Tensor::reshape(V, {B, T, num_heads, head_dim}));

  // (B, H, T, head_dim)
  Q = std::dynamic_pointer_cast<Tensor>(Tensor::transpose(Q, 1, 2));
  K = std::dynamic_pointer_cast<Tensor>(Tensor::transpose(K, 1, 2));
  V = std::dynamic_pointer_cast<Tensor>(Tensor::transpose(V, 1, 2));

  // optional RoPE (like for Qwen)
  if (rope) {
    Q = rope->forward(Q);
    K = rope->forward(K);
  }

  if (use_kv_cache) {
    if (!k_cache) {
      // First step — initialize cache from current K, V
      k_cache = K;
      v_cache = V;
    } else {
      // Subsequent steps — append new K, V to cache
      // Allocate new cache of size seq_so_far + T
      const tcapint T_cached = k_cache->shape[2];
      const tcapint T_new = T;
      const tcapint T_total = T_cached + T_new;

      TensorPtr k_new =
          Tensor::zeros({1, (tcapint)num_heads, T_total, (tcapint)head_dim});
      TensorPtr v_new =
          Tensor::zeros({1, (tcapint)num_heads, T_total, (tcapint)head_dim});

      // Copy old cache into first T_cached positions
      TensorPtr k_old_slice = Tensor::slice(k_new, 2, 0, T_cached);
      TensorPtr v_old_slice = Tensor::slice(v_new, 2, 0, T_cached);
      Weed::add_in_place(*k_old_slice, *k_cache);
      Weed::add_in_place(*v_old_slice, *v_cache);

      // Copy new K, V into remaining positions
      TensorPtr k_new_slice = Tensor::slice(k_new, 2, T_cached, T_new);
      TensorPtr v_new_slice = Tensor::slice(v_new, 2, T_cached, T_new);
      Weed::add_in_place(*k_new_slice, *K);
      Weed::add_in_place(*v_new_slice, *V);

      k_cache = k_new;
      v_cache = v_new;
    }

    // Use cache for attention
    K = k_cache;
    V = v_cache;
  }

  // scores = Q K^T
  TensorPtr Kt = Tensor::transpose(K, -2, -1);
  TensorPtr scores = Q >> Kt;

  // scale
  scores = scores / real1(std::sqrt((real1)head_dim));

  // Causal mask — only when seq_len > 1
  if (T > 1) {
    const tcapint T_q = (tcapint)T;
    const tcapint T_k = use_kv_cache ? (tcapint)K->shape[2] : T_q;
    TensorPtr mask = Tensor::zeros({T_q, T_k});
    Weed::triu_fill(*mask, mask_val);
    scores = scores + mask;
  }

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
  Serializer::write_real1_f(os, mask_val);
  Serializer::write_symint(os, d_model);
  Serializer::write_symint(os, num_heads);
  Serializer::write_symint(os, head_dim);
  Serializer::write_bool(os, use_kv_cache);
  W_q->save(os);
  W_k->save(os);
  W_v->save(os);
  W_o->save(os);
  Serializer::write_bool(os, (bool)rope);
  if (rope) {
    rope->save(os);
  }
}
} // namespace Weed
