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

#include "modules/module.hpp"

namespace Weed {
/**
 * Drop parameter values with a random probability
 */
struct Dropout : public Module {
  real1 p;
  bool training;
  TensorPtr mask;

  Dropout() : Module(DROPOUT_T) {}
  Dropout(real1 prob)
      : Module(DROPOUT_T), p(prob), training(true), mask(nullptr) {
    if ((p < ZERO_R1) || (p >= ONE_R1)) {
      throw std::invalid_argument(
          "Dropout probability must be at least 0.0 and cannot be greater than "
          "or equal to 1.0!");
    }
  }

  void train() override {
    Module::train();
    training = true;
  }
  void eval() override {
    Module::eval();
    training = false;
  }

  TensorPtr forward(const TensorPtr x) override;

  void save(std::ostream &) const;
};
typedef std::shared_ptr<Dropout> DropoutPtr;
} // namespace Weed
