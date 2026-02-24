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

#include "common/serializer.hpp"
#include "modules/module.hpp"

namespace Weed {
/**
 * Flatten axis
 */
struct Flatten : public Module {
  symint axis;
  Flatten(const symint &axis_ = -1) : Module(FLATTEN_T), axis(axis_) {}
  TensorPtr forward(const TensorPtr x) override {
    return Tensor::flatten(x, axis);
  }
  void save(std::ostream &os) const override {
    Module::save(os);
    Serializer::write_symint(os, axis);
  }
};
typedef std::shared_ptr<Flatten> FlattenPtr;
} // namespace Weed
