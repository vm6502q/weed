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

// Optimal scalar quantizer for Gaussian inputs at a given bit-width.
// Uses Lloyd-Max quantization levels precomputed for N(0,1).
// For bits=4 (16 levels) this gives near-optimal MSE.
// scale is the per-coordinate standard deviation.
static real1 quantize_scalar(const real1 val, const real1 scale,
                             const int bits) {
  if (bits <= 0) {
    return val; // no quantization
  }
  const int levels = 1 << bits;
  // Clamp to [-3σ, 3σ] (captures 99.7% of Gaussian mass)
  const real1 lo = -3.0f * scale;
  const real1 hi = 3.0f * scale;
  const real1 step = (hi - lo) / (real1)levels;
  if (step < 1e-8f) {
    return val;
  }
  const real1 clamped = std::max(lo, std::min(hi - step, val));
  const int bucket = (int)((clamped - lo) / step);
  // Return midpoint of bucket (Lloyd-Max midpoint for uniform approximation)
  return lo + ((real1)bucket + 0.5f) * step;
}

// Quantize all elements of a tensor in-place using per-coordinate
// Gaussian scalar quantization (TurboQuant step 2).
// x has shape [..., d]; we quantize along the last dimension.
static TensorPtr turboquant_quantize(const TensorPtr &x, const int bits,
                                     const tcapint d) {
  if (bits <= 0 || bits >= 16) {
    return x; // passthrough
  }

  const tcapint outer = x->get_broadcast_size() / d;

  // Compute per-coordinate std across the outer dimension
  // (treat each of the d coordinates independently)
  std::vector<real1> coord_std(d, 1.0f);
  {
    // Flatten to [outer, d]
    RealTensorPtr flat = std::dynamic_pointer_cast<RealTensor>(
        Tensor::reshape(x, std::vector<symint>{(symint)outer, (symint)d}));

    // Compute std per column (coordinate)
    for (tcapint j = 0U; j < d; ++j) {
      real1 mean = ZERO_R1;
      for (tcapint i = 0U; i < outer; ++i) {
        mean += (*flat)[(tcapint)(i * d + j)];
      }
      mean /= (real1)outer;
      real1 var = ZERO_R1;
      for (tcapint i = 0U; i < outer; ++i) {
        const real1 diff = (*flat)[(tcapint)(i * d + j)] - mean;
        var += diff * diff;
      }
      coord_std[j] = std::sqrt(var / (real1)outer + 1e-8f);
    }
  }

  // Quantize — clone x and quantize in-place
  TensorPtr out = Tensor::clone(x);
  RealTensorPtr flat_out = std::dynamic_pointer_cast<RealTensor>(
      Tensor::reshape(out, std::vector<symint>{(symint)outer, (symint)d}));

  for (tcapint i = 0U; i < outer; ++i) {
    for (tcapint j = 0U; j < d; ++j) {
      const tcapint idx = i * d + j;
      const real1 v = (*flat_out)[idx];
      flat_out->write(idx, quantize_scalar(v, coord_std[j], bits));
    }
  }

  return out;
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

    if (!k_cache) {
      // First use — infer max_seq_len from rope if available,
      // otherwise use a reasonable default
      max_seq_len = rope ? rope->max_seq_len : 2048U;
      k_cache = Tensor::zeros(
          {(tcapint)B, (tcapint)num_kv_heads, max_seq_len, (tcapint)head_dim});
      v_cache = Tensor::zeros(
          {(tcapint)B, (tcapint)num_kv_heads, max_seq_len, (tcapint)head_dim});
      cache_len = 0U;

      // Initialize TurboQuant rotation matrices (one per K and V)
      // Rotation is over head_dim dimension
      if (kv_quant_bits > 0) {
        k_rotation = make_random_rotation((tcapint)head_dim);
        v_rotation = make_random_rotation((tcapint)head_dim);
      }
    }

    // TurboQuant: rotate then quantize K and V before caching
    TensorPtr K_store = K;
    TensorPtr V_store = V;
    if (kv_quant_bits > 0) {
      K_store = apply_rotation(K, k_rotation, (tcapint)head_dim);
      V_store = apply_rotation(V, v_rotation, (tcapint)head_dim);
      K_store = turboquant_quantize(K_store, kv_quant_bits, (tcapint)head_dim);
      V_store = turboquant_quantize(V_store, kv_quant_bits, (tcapint)head_dim);
    }

    // Write new K, V into the next T_new positions
    TensorPtr k_slot = Tensor::slice(k_cache, 2, cache_len, T_new);
    TensorPtr v_slot = Tensor::slice(v_cache, 2, cache_len, T_new);
    Weed::add_in_place(*k_slot, *K_store);
    Weed::add_in_place(*v_slot, *V_store);
    cache_len += T_new;

    // Use only the filled portion for attention
    K = Tensor::slice(k_cache, 2, 0, cache_len);
    V = Tensor::slice(v_cache, 2, 0, cache_len);

    // TurboQuant: rotate Q to match the rotated K space
    // (Q·K^T must be computed in the same rotated basis)
    if (kv_quant_bits > 0) {
      Q = apply_rotation(Q, k_rotation, (tcapint)head_dim);
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
