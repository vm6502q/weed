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
 * Softmax activation
 */
struct Softmax : public Module {
  tcapint axis;

  explicit Softmax(const tcapint &axis_ = -1) : Module(SOFTMAX), axis(axis_) {}

  TensorPtr forward(const TensorPtr x) override;
  void save(std::ostream &) const override;
};
typedef std::shared_ptr<Softmax> SoftmaxPtr;
} // namespace Weed
