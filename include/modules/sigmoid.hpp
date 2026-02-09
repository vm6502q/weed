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
 * Convenience wrapper on sigmoid as a module
 */
struct Sigmoid : public Module {
  Sigmoid() : Module(SIGMOID_T) {}
  TensorPtr forward(const TensorPtr x) override { return Tensor::sigmoid(x); }
};
typedef std::shared_ptr<Sigmoid> SigmoidPtr;
} // namespace Weed
