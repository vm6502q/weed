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
 * Gated recurrent unit
 */
struct GRU : public Module {
  tcapint input_dim;
  tcapint hidden_dim;

  LinearPtr W_x; // x → 3H
  LinearPtr W_h; // h → 3H

  TensorPtr state;

  GRU() : Module(GRU_T) {}
  GRU(tcapint in, tcapint hid, DeviceTag dtag = DeviceTag::DEFAULT_DEVICE)
      : Module(GRU_T), input_dim(in), hidden_dim(hid),
        W_x(std::make_shared<Linear>(in, 3 * hid, true, DType::REAL, dtag)),
        W_h(std::make_shared<Linear>(hid, 3 * hid, true, DType::REAL, dtag)),
        state(Tensor::zeros({hidden_dim})) {}

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

  TensorPtr forward(const TensorPtr) override;

  void save(std::ostream &) const override;
};
typedef std::shared_ptr<GRU> GRUPtr;
} // namespace Weed
