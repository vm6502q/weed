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
 * Convenience wrapper on GeLU as a module
 */
struct GeLU : public Module {
  GeLU() : Module(GELU_T) {}
  TensorPtr forward(const TensorPtr x) override { return Tensor::gelu(x); }
};
typedef std::shared_ptr<GeLU> GeLUPtr;
} // namespace Weed
