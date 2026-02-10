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
 * Positional encoding for transformers (constant operation, no gradient effect)
 */
struct PositionalEncoding : public Module {
  Tanh() : Module(POSITIONAL_ENCODING_T) {}
  TensorPtr forward(const TensorPtr x) {
    // x: [B, T, D]
    TensorPtr pe_slice = pe->slice(0, 0, x->shape[1]);
    return x + pe_slice;
  }
};
typedef std::shared_ptr<Tanh> TanhPtr;
} // namespace Weed
