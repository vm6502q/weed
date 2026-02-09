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

#include "modules/layernorm.hpp"
#include "modules/linear.hpp"
#include "modules/module.hpp"
#include "modules/multihead_attention.hpp"
#include "modules/relu.hpp"

namespace Weed {
struct TransformerEncoderLayer : public Module {
  tcapint d_model;
  tcapint d_ff;
  tcapint num_heads;

  MultiHeadAttentionPtr self_attn;

  LinearPtr ff1;
  LinearPtr ff2;

  LayerNormPtr norm1;
  LayerNormPtr norm2;

  ReLUPtr activation;

  std::vector<ParameterPtr> param_vector;

  TransformerEncoderLayer(tcapint d_model_, tcapint num_heads_, tcapint d_ff_,
                          DeviceTag dtag = DEFAULT_DEVICE)
      : Module(TRANSFORMER_ENCODER_LAYER_T), d_model(d_model_), d_ff(d_ff_),
        num_heads(num_heads_), self_attn(std::make_shared<MultiHeadAttention>(
                                   d_model_, num_heads_, dtag)),
        ff1(std::make_shared<Linear>(d_model_, d_ff_, true, DType::REAL, dtag)),
        ff2(std::make_shared<Linear>(d_ff_, d_model_, true, DType::REAL, dtag)),
        norm1(std::make_shared<LayerNorm>(d_model_, dtag)),
        norm2(std::make_shared<LayerNorm>(d_model_, dtag)),
        activation(std::make_shared<ReLU>()) {
    param_vector = self_attn->parameters();
    auto add = [&](const std::vector<ParameterPtr> &q) {
      param_vector.insert(param_vector.end(), q.begin(), q.end());
    };
    add(ff1->parameters());
    add(ff2->parameters());
    add(norm1->parameters());
    add(norm2->parameters());
  }

  std::vector<ParameterPtr> parameters() override { return param_vector; }

  void train() override {
    self_attn->train();
    ff1->train();
    ff2->train();
    norm1->train();
    norm2->train();
    activation->train();
  }
  void eval() override {
    self_attn->eval();
    ff1->eval();
    ff2->eval();
    norm1->eval();
    norm2->eval();
    activation->eval();
  }

  TensorPtr forward(const TensorPtr x) override;

  void save(std::ostream &) const override;
};
typedef std::shared_ptr<TransformerEncoderLayer> TransformerEncoderLayerPtr;
} // namespace Weed
