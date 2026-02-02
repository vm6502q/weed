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
#include "tensors/parameter.hpp"

namespace Weed {
/**
 * Standard interface for sequential models of multiple layers
 */
struct Sequential : public Module {
  std::vector<ModulePtr> layers;
  std::vector<ParameterPtr> param_vector;

  Sequential(const std::vector<ModulePtr>& l) : layers(l), param_vector() {
    for (size_t i = 0U; i < layers.size(); ++i) {
      const std::vector<ParameterPtr> l = layers[i]->parameters();
      param_vector.insert(param_vector.end(), l.begin(), l.end());
    }
  }

  TensorPtr forward(TensorPtr x) override {
    TensorPtr tmp = x;
    for (size_t i = 0U; i < layers.size(); ++i) {
      tmp = layers[i]->forward(x);
    }

    return tmp;
  }

  std::vector<ParameterPtr> parameters() override { return param_vector; }
};
} // namespace Weed
