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

#include "modules/transformer_encoder_layer.hpp"
#include "common/serializer.hpp"

#include "modules/gelu.hpp"
#include "modules/relu.hpp"
#include "modules/sigmoid.hpp"
#include "modules/tanh.hpp"

namespace Weed {
TransformerEncoderLayer::TransformerEncoderLayer(
    const tcapint &d_model_, const tcapint &num_heads_, const tcapint &d_ff_,
    const DeviceTag &dtag, const ActivationFunctionType &afn)
    : Module(TRANSFORMER_ENCODER_LAYER_T), d_model(d_model_), d_ff(d_ff_),
      num_heads(num_heads_), self_attn(std::make_shared<MultiHeadAttention>(
                                 d_model_, num_heads_, dtag)),
      ff1(std::make_shared<Linear>(d_model_, d_ff_, true, true, DType::REAL,
                                   dtag)),
      ff2(std::make_shared<Linear>(d_ff_, d_model_, true, true, DType::REAL,
                                   dtag)),
      norm1(std::make_shared<LayerNorm>(d_model_, dtag)),
      norm2(std::make_shared<LayerNorm>(d_model_, dtag)) {
  switch (afn) {
  case SIGMOID_FN:
    activation = std::make_shared<Sigmoid>();
    break;
  case TANH_FN:
    activation = std::make_shared<Tanh>();
    break;
  case RELU_FN:
    activation = std::make_shared<ReLU>();
    break;
  case GELU_FN:
  default:
    activation = std::make_shared<GeLU>();
  }

  param_vector = self_attn->parameters();
  auto add = [&](const std::vector<ParameterPtr> &q) {
    param_vector.insert(param_vector.end(), q.begin(), q.end());
  };
  add(ff1->parameters());
  add(ff2->parameters());
  add(norm1->parameters());
  add(norm2->parameters());
}
TensorPtr TransformerEncoderLayer::forward(const TensorPtr x) {
  // Self-attention block
  TensorPtr attn_out = self_attn->forward(x);
  TensorPtr x1 = norm1->forward(x + attn_out);

  // Feed-forward block
  TensorPtr ff = ff1->forward(x1);
  ff = activation->forward(ff);
  ff = ff2->forward(ff);

  TensorPtr out = norm2->forward(x1 + ff);

  return out;
}
void TransformerEncoderLayer::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, d_model);
  Serializer::write_tcapint(os, d_ff);
  Serializer::write_tcapint(os, num_heads);
  self_attn->save(os);
  ff1->save(os);
  ff2->save(os);
  norm1->save(os);
  norm2->save(os);
  activation->save(os);
}
} // namespace Weed
