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
 * Max activation
 */
struct Max : public Module {
  symint axis;
  Max(const symint &axis_ = -1) : Module(MAX_T), axis(axis_) {}
  TensorPtr forward(const TensorPtr x) override { return Tensor::max(x, axis); }
  void save(std::ostream &os) const override {
    Serializer::write_symint(os, axis);
  }
};
typedef std::shared_ptr<Max> MaxPtr;
} // namespace Weed
