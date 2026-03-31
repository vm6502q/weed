//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of
// Qrack.
//
// KV cache quantization based on TurboQuant (Zandieh et al., arXiv:2504.19874),
// Apache 2.0 open-source implementation by TheTom
// (github.com/TheTom/turboquant_plus), and (Anthropic) Claude.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#include "modules/multihead_attention.hpp"
#include "common/serializer.hpp"
#include "ops/in_place.hpp"
#include "ops/triu_fill.hpp"
#include "tensors/real_tensor.hpp"

#include <cmath>
#include <random>

namespace Weed {

// ---------------------------------------------------------------------------
// TurboQuant helpers
// ---------------------------------------------------------------------------

void QuantizedKVCache::allocate(const tcapint max_outer_, const tcapint d_, const int bits_) {
    d = d_;
    bits = bits_;
    max_outer = max_outer_;
    outer = 0U;
    scales.resize(d_, 1.0f);
    const int vpw = values_per_word();
    // Each row of d values needs ceil(d / vpw) words
    const tcapint words_per_row = (d + vpw - 1) / vpw;
    packed.assign(max_outer_ * words_per_row, 0);
}

void QuantizedKVCache::write_row(const tcapint row, const real1 *vals) {
    const int vpw = values_per_word();
    const int levels = 1 << bits;
    const tcapint words_per_row = ((tcapint)d + vpw - 1) / vpw;
    const tcapint base = row * words_per_row;
    for (tcapint j = 0U; j < (tcapint)d; ++j) {
        // Quantize to bucket index
        const real1 lo = -3.0f * scales[j];
        const real1 hi =  3.0f * scales[j];
        const real1 step = (hi - lo) / (real1)levels;
        int bucket = 0;
        if (step > 1e-8f) {
            const real1 clamped = std::max(lo, std::min(hi - step, vals[j]));
            bucket = (int)((clamped - lo) / step);
            bucket = std::max(0, std::min(levels - 1, bucket));
        }
        // Pack into word
        const tcapint word_idx = base + j / vpw;
        const int bit_offset = (j % vpw) * bits;
        packed[word_idx] |= ((symint)bucket << bit_offset);
    }
}

void QuantizedKVCache::read_row(const tcapint row, real1 *vals) const {
    const int vpw = values_per_word();
    const int levels = 1 << bits;
    const int mask = levels - 1;
    const tcapint words_per_row = ((tcapint)d + vpw - 1) / vpw;
    const tcapint base = row * words_per_row;
    for (tcapint j = 0U; j < (tcapint)d; ++j) {
        const tcapint word_idx = base + j / vpw;
        const int bit_offset = (j % vpw) * bits;
        const int bucket = (int)((packed[word_idx] >> bit_offset) & mask);
        // Dequantize: midpoint of bucket
        const real1 lo = -3.0f * scales[j];
        const real1 hi =  3.0f * scales[j];
        const real1 step = (hi - lo) / (real1)levels;
        vals[j] = lo + ((real1)bucket + 0.5f) * step;
    }
}

// Build a random orthogonal rotation matrix of size d×d using
// Householder reflections (a simple QR via random Gaussian matrix).
// This is the "random rotation" step of TurboQuant.
static std::vector<real1> make_random_rotation(const tcapint d) {
  std::mt19937 rng(std::random_device{}());
  std::normal_distribution<real1> normal(0.0f, 1.0f);

  // Fill d×d matrix with iid Gaussians (column-major)
  std::vector<real1> R(d * d);
  for (auto &v : R) {
    v = normal(rng);
  }

  // Gram-Schmidt orthogonalization (column-major: column j starts at R[j*d])
  for (tcapint j = 0U; j < d; ++j) {
    // Normalize column j
    real1 norm = ZERO_R1;
    for (tcapint i = 0U; i < d; ++i) {
      norm += R[j * d + i] * R[j * d + i];
    }
    norm = std::sqrt(norm);
    if (norm < 1e-8f) {
      norm = 1e-8f;
    }
    for (tcapint i = 0U; i < d; ++i) {
      R[j * d + i] /= norm;
    }
    // Subtract projection onto column j from all subsequent columns
    for (tcapint k = j + 1U; k < d; ++k) {
      real1 dot = ZERO_R1;
      for (tcapint i = 0U; i < d; ++i) {
        dot += R[j * d + i] * R[k * d + i];
      }
      for (tcapint i = 0U; i < d; ++i) {
        R[k * d + i] -= dot * R[j * d + i];
      }
    }
  }

  return R;
}

// Apply rotation R (d×d, column-major) to the last dimension of x.
// x has shape [..., d]; result has same shape.
static TensorPtr apply_rotation(const TensorPtr &x, const std::vector<real1> &R,
                                const tcapint d) {
  // Flatten x to [..., d], matmul with R, reshape back.
  // R as a Tensor: shape [d, d], column-major (already in our format).
  const std::vector<tcapint> x_shape = x->shape;
  const tcapint outer = x->get_broadcast_size() / d;

  TensorPtr x_flat =
      Tensor::reshape(x, std::vector<symint>{(symint)outer, (symint)d});
  TensorPtr R_tensor = std::make_shared<Tensor>(R, std::vector<tcapint>{d, d});
  TensorPtr y_flat = x_flat >> R_tensor;

  // Reshape back to original shape
  std::vector<symint> out_shape(x_shape.begin(), x_shape.end());
  return Tensor::reshape(y_flat, out_shape);
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------
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

    if (!k_cache && !k_qcache.d) {
        max_seq_len = rope ? rope->max_seq_len : 2048U;
        cache_len = 0U;

        if (kv_quant_bits > 0) {
            k_rotation = make_random_rotation((tcapint)head_dim);
            v_rotation = make_random_rotation((tcapint)head_dim);

            // Pre-allocate packed caches
            // outer = B * num_kv_heads * max_seq_len rows of head_dim values
            const tcapint max_rows =
                (tcapint)B * (tcapint)num_kv_heads * max_seq_len;
            k_qcache.allocate(max_rows, (tcapint)head_dim, kv_quant_bits);
            v_qcache.allocate(max_rows, (tcapint)head_dim, kv_quant_bits);
        } else {
            k_cache = Tensor::zeros({(tcapint)B, (tcapint)num_kv_heads,
                                      max_seq_len, (tcapint)head_dim});
            v_cache = Tensor::zeros({(tcapint)B, (tcapint)num_kv_heads,
                                      max_seq_len, (tcapint)head_dim});
        }
    }

    // Rotate K and V before storing
    TensorPtr K_store = K;
    TensorPtr V_store = V;
    if (kv_quant_bits > 0) {
        K_store = apply_rotation(K, k_rotation, (tcapint)head_dim);
        V_store = apply_rotation(V, v_rotation, (tcapint)head_dim);
    }

    if (kv_quant_bits > 0) {
        // Write new tokens into packed cache
        // K_store shape: [B, num_kv_heads, T_new, head_dim]
        // Iterate over B * num_kv_heads * T_new rows
        TensorPtr K_flat_ptr = Tensor::reshape(K_store,
                std::vector<symint>{(symint)((tcapint)B *
                    (tcapint)num_kv_heads * T_new), (symint)head_dim});
        TensorPtr V_flat_ptr = Tensor::reshape(V_store,
                std::vector<symint>{(symint)((tcapint)B *
                    (tcapint)num_kv_heads * T_new), (symint)head_dim});
        RealTensor *K_flat = static_cast<RealTensor *>(K_flat_ptr.get());
        RealTensor *V_flat = static_cast<RealTensor *>(V_flat_ptr.get());

        // Compute scales from this batch if first write
        if (cache_len == 0U) {
            const tcapint n_rows = (tcapint)B * (tcapint)num_kv_heads * T_new;
            for (tcapint j = 0U; j < (tcapint)head_dim; ++j) {
                real1 var = ZERO_R1;
                for (tcapint i = 0U; i < n_rows; ++i) {
                    const real1 v = (*K_flat)[i * (tcapint)head_dim + j];
                    var += v * v;
                }
                k_qcache.scales[j] = std::sqrt(var / (real1)n_rows + 1e-8f);
                var = ZERO_R1;
                for (tcapint i = 0U; i < n_rows; ++i) {
                    const real1 v = (*V_flat)[i * (tcapint)head_dim + j];
                    var += v * v;
                }
                v_qcache.scales[j] = std::sqrt(var / (real1)n_rows + 1e-8f);
            }
        }

        // Write rows into packed cache
        const tcapint base_row =
            cache_len * (tcapint)B * (tcapint)num_kv_heads;
        const tcapint n_rows = (tcapint)B * (tcapint)num_kv_heads * T_new;
        std::vector<real1> row_buf((tcapint)head_dim);
        for (tcapint i = 0U; i < n_rows; ++i) {
            for (tcapint j = 0U; j < (tcapint)head_dim; ++j) {
                row_buf[j] = (*K_flat)[i * (tcapint)head_dim + j];
            }
            k_qcache.write_row(base_row + i, row_buf.data());
            for (tcapint j = 0U; j < (tcapint)head_dim; ++j) {
                row_buf[j] = (*V_flat)[i * (tcapint)head_dim + j];
            }
            v_qcache.write_row(base_row + i, row_buf.data());
        }
        cache_len += T_new;

        // Reconstruct K and V tensors from packed cache for attention
        const tcapint total_rows =
            cache_len * (tcapint)B * (tcapint)num_kv_heads;
        std::vector<real1> k_data(total_rows * (tcapint)head_dim);
        std::vector<real1> v_data(total_rows * (tcapint)head_dim);
        for (tcapint i = 0U; i < total_rows; ++i) {
            k_qcache.read_row(i, &k_data[i * (tcapint)head_dim]);
            v_qcache.read_row(i, &v_data[i * (tcapint)head_dim]);
        }

        K = std::make_shared<Tensor>(k_data,
            std::vector<tcapint>{(tcapint)B, (tcapint)num_kv_heads,
                                  cache_len, (tcapint)head_dim});
        V = std::make_shared<Tensor>(v_data,
            std::vector<tcapint>{(tcapint)B, (tcapint)num_kv_heads,
                                  cache_len, (tcapint)head_dim});

        // Rotate Q to match rotated K basis
        Q = apply_rotation(Q, k_rotation, (tcapint)head_dim);

    } else {
        // Unquantized path — unchanged
        TensorPtr k_slot = Tensor::slice(k_cache, 2, cache_len, T_new);
        TensorPtr v_slot = Tensor::slice(v_cache, 2, cache_len, T_new);
        Weed::add_in_place(*k_slot, *K_store);
        Weed::add_in_place(*v_slot, *V_store);
        cache_len += T_new;
        K = Tensor::slice(k_cache, 2, 0, cache_len);
        V = Tensor::slice(v_cache, 2, 0, cache_len);
    }
  }

  // GQA: broadcast K and V from kv_heads to num_heads
  if (num_kv_heads < num_heads) {
    const symint groups = num_heads / num_kv_heads;
    const tcapint T_k = (tcapint)K->shape[2];

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

  // TurboQuant: rotate output back from V's rotated basis
  if (use_kv_cache && kv_quant_bits > 0) {
    out = apply_rotation(out, v_rotation, (tcapint)head_dim);
  }

  // (B, T, num_heads * head_dim)
  const symint attn_dim = (symint)num_heads * (symint)head_dim;
  out = Tensor::reshape(out, {B, T, attn_dim});
  out = W_o->forward(out);

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
  Serializer::write_symint(os, (symint)kv_quant_bits);
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
