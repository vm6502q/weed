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

  Q = Tensor::reshape(Q, std::vector<symint>{B, T, num_heads, head_dim});
  K = Tensor::reshape(K, std::vector<symint>{B, T, num_kv_heads, head_dim});
  V = Tensor::reshape(V, std::vector<symint>{B, T, num_kv_heads, head_dim});

  Q = Tensor::transpose(Q, 1, 2); // [B, num_heads,  T, head_dim]
  K = Tensor::transpose(K, 1, 2); // [B, kv_heads,   T, head_dim]
  V = Tensor::transpose(V, 1, 2); // [B, kv_heads,   T, head_dim]

  // optional RoPE (like for Qwen)
  if (rope) {
    Q = rope->forward(Q);
    K = rope->forward(K);
  }

  if (use_kv_cache) {
    const tcapint T_new = (tcapint)T;

    if (!k_cache) {
        // First use — infer max_seq_len from rope if available,
        // otherwise use a reasonable default
        max_seq_len = rope ? rope->max_seq_len : 2048U;
        k_cache = Tensor::zeros({(tcapint)B, (tcapint)num_kv_heads,
                                  max_seq_len, (tcapint)head_dim});
        v_cache = Tensor::zeros({(tcapint)B, (tcapint)num_kv_heads,
                                  max_seq_len, (tcapint)head_dim});
        cache_len = 0U;
    }

    // Write new K, V into the next T_new positions
    TensorPtr k_slot = Tensor::slice(k_cache, 2, cache_len, T_new);
    TensorPtr v_slot = Tensor::slice(v_cache, 2, cache_len, T_new);
    Weed::add_in_place(*k_slot, *K);
    Weed::add_in_place(*v_slot, *V);
    cache_len += T_new;

    // Use only the filled portion for attention
    K = Tensor::slice(k_cache, 2, 0, cache_len);
    V = Tensor::slice(v_cache, 2, 0, cache_len);
  }

  // GQA: broadcast K and V from kv_heads to num_heads
  if (num_kv_heads < num_heads) {
    const symint groups = num_heads / num_kv_heads;
    const tcapint T_k = (tcapint)K->shape[2]; // ← use actual K length, not T

    TensorPtr K_rep =
        Tensor::zeros({(tcapint)B, (tcapint)num_heads, T_k, (tcapint)head_dim});
    TensorPtr V_rep =
        Tensor::zeros({(tcapint)B, (tcapint)num_heads, T_k, (tcapint)head_dim});

    for (symint g = 0; g < groups; ++g) {
      TensorPtr K_slice = Tensor::slice(K_rep, 1, (tcapint)(g * num_kv_heads),
                                        (tcapint)num_kv_heads);
      TensorPtr V_slice = Tensor::slice(V_rep, 1, (tcapint)(g * num_kv_heads),
                                        (tcapint)num_kv_heads);
      Weed::add_in_place(*K_slice, *K);
      Weed::add_in_place(*V_slice, *V);
    }

    K = K_rep;
    V = V_rep;
  }

  // scores = Q K^T
  TensorPtr Kt = Tensor::transpose(K, -2, -1);
  TensorPtr scores = Q >> Kt;
  Q = nullptr;

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

  K = nullptr;
  Kt = nullptr;

  // softmax over last axis
  TensorPtr weights = Tensor::softmax(scores, -1);

  // out = weights V
  TensorPtr out = weights >> V;

  V = nullptr;
  weights = nullptr;

  // (B, T, H, head_dim)
  out = Tensor::transpose(out, 1, 2);

  // (B, T, num_heads * head_dim)
  const symint attn_dim = (symint)num_heads * (symint)head_dim;
  out = Tensor::reshape(out, {B, T, attn_dim});
  out = W_o->forward(out);

  // final projection: attn_dim → d_model
  return out;
}

void MultiHeadAttention::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_real1_f(os, mask_val);
  Serializer::write_symint(os, d_model);
  Serializer::write_symint(os, num_heads);
  Serializer::write_symint(os, num_kv_heads);
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
