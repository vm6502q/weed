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

#include "enums/activation_function_type.hpp"
#include "modules/multihead_attention.hpp"
#include "modules/rms_norm.hpp"
#include "modules/rope.hpp"
#include "modules/swiglu.hpp"

namespace Weed {
/**
 * Qwen-style model decoder layer (contributed by Anthropic Claude)
 */
struct QwenDecoderLayer : public Module {
  tcapint d_model;
  tcapint num_heads;
  tcapint num_kv_heads;
  MultiHeadAttentionPtr self_attn;
  RoPEPtr rope;
  SwiGLUPtr mlp;
  RMSNormPtr input_layernorm;
  RMSNormPtr post_attention_layernorm;

  std::vector<ParameterPtr> param_vector;

  QwenDecoderLayer() : Module(QWEN_DECODER_LAYER_T) {}
  QwenDecoderLayer(const tcapint &d_model_, const tcapint &num_heads_,
                   const tcapint &num_kv_heads_, const tcapint &d_ff_,
                   const tcapint &max_seq_len = 2048U,
                   const real1_f &rope_base = 10000.0f,
                   const real1_f &eps = 1e-6f, const int64_t &did = -1)
      : Module(QWEN_DECODER_LAYER_T), d_model(d_model_), num_heads(num_heads_),
        num_kv_heads(num_kv_heads_) {
    const tcapint head_dim = d_model_ / num_heads_;
    rope = std::make_shared<RoPE>(head_dim, max_seq_len, rope_base);
    self_attn = std::make_shared<MultiHeadAttention>(
        d_model_, num_heads_, head_dim, DEFAULT_DEVICE, rope, ZERO_R1, did);
    mlp = std::make_shared<SwiGLU>(d_model_, d_ff_);
    input_layernorm = std::make_shared<RMSNorm>(d_model_, -1);
    post_attention_layernorm = std::make_shared<RMSNorm>(d_model_, -1);
  }

  void _register_params() {
    auto p = self_attn->parameters();
    auto m = mlp->parameters();
    auto w1 = input_layernorm->parameters();
    auto w2 = post_attention_layernorm->parameters();
    param_vector.insert(param_vector.end(), p.begin(), p.end());
    param_vector.insert(param_vector.end(), m.begin(), m.end());
    param_vector.insert(param_vector.end(), w1.begin(), w1.end());
    param_vector.insert(param_vector.end(), w2.begin(), w2.end());
  }

  std::vector<ParameterPtr> parameters() override { return param_vector; }

  TensorPtr forward(const TensorPtr x) override {
    // Pre-norm + attention + residual
    TensorPtr residual = x;
    TensorPtr h = self_attn->forward(input_layernorm->forward(x));
    h = h + residual;

    // Pre-norm + FFN + residual
    residual = h;
    h = mlp->forward(post_attention_layernorm->forward(h));

    return h + residual;
  }

  void save(std::ostream &os) const override;
};
typedef std::shared_ptr<QwenDecoderLayer> QwenDecoderLayerPtr;
} // namespace Weed
