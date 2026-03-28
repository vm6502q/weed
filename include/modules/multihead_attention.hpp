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

#include "modules/linear.hpp"
#include "modules/rope.hpp"

namespace Weed {
/**
 * Attention mechanism used by transformer models
 */
struct MultiHeadAttention : public Module {
  symint d_model;
  symint num_heads;
  symint num_kv_heads;
  symint head_dim;
  real1_f mask_val;

  LinearPtr W_q;
  LinearPtr W_k;
  LinearPtr W_v;
  LinearPtr W_o;

  RoPEPtr rope;

  bool use_kv_cache;
  TensorPtr k_cache; // [1, num_heads, seq_so_far, head_dim]
  TensorPtr v_cache; // [1, num_heads, seq_so_far, head_dim]

  std::vector<ParameterPtr> param_vector;

  MultiHeadAttention() : Module(MULTIHEAD_ATTENTION_T) {}
  MultiHeadAttention(tcapint d_model_, tcapint num_heads_,
                     tcapint num_kv_heads_ = 0, tcapint head_dim_ = 0U,
                     DeviceTag dtag = DEFAULT_DEVICE, RoPEPtr r = nullptr,
                     real1_f mask_val_ = ZERO_R1, const int64_t did = -1,
                     const bool _use_kv_cache = true)
      : Module(MULTIHEAD_ATTENTION_T), d_model(d_model_), num_heads(num_heads_),
        num_kv_heads(num_kv_heads_ ? num_kv_heads_ : num_heads),
        head_dim(!head_dim_ ? d_model_ / num_heads_ : head_dim_),
        mask_val(mask_val_),
        W_q(std::make_shared<Linear>(d_model_, d_model_, true, true,
                                     DType::REAL, dtag, did)),
        W_k(std::make_shared<Linear>(d_model_, d_model_, true, true,
                                     DType::REAL, dtag, did)),
        W_v(std::make_shared<Linear>(d_model_, d_model_, true, true,
                                     DType::REAL, dtag, did)),
        W_o(std::make_shared<Linear>(d_model_, d_model_, true, true,
                                     DType::REAL, dtag, did)),
        rope(r), use_kv_cache(_use_kv_cache) {
    if (d_model % num_heads) {
      throw std::invalid_argument("d_model must be divisible by num_heads");
    }

    param_vector = W_q->parameters();
    auto add = [&](const std::vector<ParameterPtr> &q) {
      param_vector.insert(param_vector.end(), q.begin(), q.end());
    };
    add(W_k->parameters());
    add(W_v->parameters());
    add(W_o->parameters());

    if (mask_val == ZERO_R1) {
#if WEED_FPPOW > 5
      mask_val = -8.988465674e307; // -2^1023
#elif WEED_FPPOW > 4
      mask_val = -1.701411835e38; // -2^127
#else
      mask_val = -65536; // -2^16
#endif
    }
  }

  std::vector<ParameterPtr> parameters() override { return param_vector; }

  void train() override {
    W_q->train();
    W_k->train();
    W_v->train();
    W_o->train();
  }
  void eval() override {
    W_q->eval();
    W_k->eval();
    W_v->eval();
    W_o->eval();
  }

  void reset_cache() override {
    k_cache = nullptr;
    v_cache = nullptr;
  }

  TensorPtr forward(const TensorPtr x) override;

  void save(std::ostream &) const override;
};
typedef std::shared_ptr<MultiHeadAttention> MultiHeadAttentionPtr;
} // namespace Weed
