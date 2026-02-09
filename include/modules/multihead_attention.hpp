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
#include "modules/module.hpp"

namespace Weed {
struct MultiHeadAttention : public Module {
  tcapint d_model;
  tcapint num_heads;
  tcapint head_dim;

  LinearPtr W_q;
  LinearPtr W_k;
  LinearPtr W_v;
  LinearPtr W_o;

  std::vector<ParameterPtr> param_vector;

  MultiHeadAttention() : Module(MULTIHEAD_ATTENTION_T) {}
  MultiHeadAttention(tcapint d_model_, tcapint num_heads_,
                     DeviceTag dtag = DEFAULT_DEVICE)
      : Module(MULTIHEAD_ATTENTION_T), d_model(d_model_), num_heads(num_heads_),
        head_dim(d_model_ / num_heads_),
        W_q(std::make_shared<Linear>(d_model_, d_model_, true, DType::REAL,
                                     dtag)),
        W_k(std::make_shared<Linear>(d_model_, d_model_, true, DType::REAL,
                                     dtag)),
        W_v(std::make_shared<Linear>(d_model_, d_model_, true, DType::REAL,
                                     dtag)),
        W_o(std::make_shared<Linear>(d_model_, d_model_, true, DType::REAL,
                                     dtag)) {
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

  TensorPtr forward(const TensorPtr x) override;

  void save(std::ostream &) const override;
};
typedef std::shared_ptr<MultiHeadAttention> MultiHeadAttentionPtr;
} // namespace Weed
