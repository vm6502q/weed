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

  Linear W_x; // input -> 4H
  Linear W_h; // hidden -> 4H

  std::vector<LSTMState> state;

  LSTM(tcapint in, tcapint hid, DeviceTag dtag = DEFAULT_DEVICE)
      : Module(LSTM_T), input_dim(in), hidden_dim(hid),
        W_x(in, 4 * hid, true, DType::REAL, dtag),
        W_h(hid, 4 * hid, true, DType::REAL, dtag),
        state{LSTMState{Tensor::zeros(std::vector<tcapint>{hidden_dim}),
                        Tensor::zeros(std::vector<tcapint>{hidden_dim})}} {}

  std::vector<ParameterPtr> parameters() override {
    auto px = W_x.parameters();
    auto ph = W_h.parameters();
    px.insert(px.end(), ph.begin(), ph.end());
    return px;
  }

  TensorPtr forward(const TensorPtr) override;
};
} // namespace Weed
