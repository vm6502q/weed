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

namespace Weed {
/**
 * Long short-term memory state
 */
struct LSTMState {
  TensorPtr h;
  TensorPtr c;
};

/**
 * Long short-term memory
 */
struct LSTM : public Module {
  // Contributed by Elara (the OpenAI custom GPT)

  tcapint input_dim;
  tcapint hidden_dim;

  LinearPtr W_x; // input -> 4H
  LinearPtr W_h; // hidden -> 4H

  LSTMState state;

  LSTM() : Module(LSTM_T) {}
  LSTM(tcapint in, tcapint hid, DeviceTag dtag = DEFAULT_DEVICE,
       const int64_t &did = -1)
      : Module(LSTM_T), input_dim(in), hidden_dim(hid),
        W_x(std::make_shared<Linear>(in, 4 * hid, true, true, DType::REAL, dtag,
                                     did)),
        W_h(std::make_shared<Linear>(hid, 4 * hid, true, true, DType::REAL,
                                     dtag, did)),
        state{Tensor::zeros(std::vector<tcapint>{hidden_dim}),
              Tensor::zeros(std::vector<tcapint>{hidden_dim})} {}

  std::vector<ParameterPtr> parameters() override {
    auto px = W_x->parameters();
    auto ph = W_h->parameters();
    px.insert(px.end(), ph.begin(), ph.end());
    return px;
  }

  void train() override {
    W_x->train();
    W_h->train();
  }
  void eval() override {
    W_x->eval();
    W_h->eval();
  }

  void migrate_cpu() override {
    W_x->migrate_cpu();
    W_h->migrate_cpu();
  }
  void migrate_gpu() override {
    W_x->migrate_gpu();
    W_h->migrate_gpu();
  }

  TensorPtr forward(const TensorPtr) override;

  void save(std::ostream &) const override;
};
typedef std::shared_ptr<LSTM> LSTMPtr;
} // namespace Weed
