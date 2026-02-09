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
#include "modules/layernorm.hpp"
#include "modules/linear.hpp"
#include "modules/module.hpp"
#include "modules/multihead_attention.hpp"

namespace Weed {
/**
 * Single layer of a transformer model, which can be composed with Sequential to
 * build deep transformers
 */
struct TransformerEncoderLayer : public Module {
  tcapint d_model;
  tcapint d_ff;
  tcapint num_heads;

  MultiHeadAttentionPtr self_attn;

  LinearPtr ff1;
  LinearPtr ff2;

  LayerNormPtr norm1;
  LayerNormPtr norm2;

  ModulePtr activation;

  std::vector<ParameterPtr> param_vector;

  TransformerEncoderLayer() : Module(TRANSFORMER_ENCODER_LAYER_T) {}
  TransformerEncoderLayer(const tcapint &d_model_, const tcapint &num_heads_,
                          const tcapint &d_ff_,
                          const DeviceTag &dtag = DEFAULT_DEVICE,
                          const ActivationFunctionType &afn = GELU_FN);

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
