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
 * Convenience wrapper on reshape as a module
 */
struct Reshape : public Module {
  std::vector<symint> shape;
  Reshape(const std::vector<symint>& s) : Module(RESHAPE_T), shape(s) {}
  TensorPtr forward(const TensorPtr x) override { return Tensor::reshape(x, shape); }
  void save(std::ostream &os) const override {
    Serializer::write_tcapint(os, (tcapint)shape.size());
    for (size_t i = 0U; i < shape.size(); ++i) {
      Serializer::write_symint(os, shape[i]);
    }
  }
};
typedef std::shared_ptr<Reshape> ReshapePtr;
} // namespace Weed
