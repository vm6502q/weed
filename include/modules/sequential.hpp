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
 * Standard interface for sequential models of multiple layers
 */
struct Sequential : public Module {
  std::vector<ModulePtr> layers;
  std::vector<ParameterPtr> param_vector;

  Sequential(const std::vector<ModulePtr> &l)
      : Module(SEQUENTIAL_T), layers(l), param_vector() {
    for (size_t i = 0U; i < layers.size(); ++i) {
      const std::vector<ParameterPtr> p = layers[i]->parameters();
      param_vector.insert(param_vector.end(), p.begin(), p.end());
    }
  }

  void train() override {
    for (const ModulePtr &m : layers) {
      m->train();
    }
  }
  void eval() override {
    for (const ModulePtr &m : layers) {
      m->eval();
    }
  }

  using Module::forward;
  TensorPtr forward(const TensorPtr x) override {
    TensorPtr tmp = x;
    for (size_t i = 0U; i < layers.size(); ++i) {
      tmp = layers[i]->forward(tmp);
    }

    return tmp;
  }
  TensorPtr forward(const SymbolTensorPtr x) {
    if (layers.empty()) {
      return std::make_shared<Tensor>();
    }
    TensorPtr tmp = layers[0]->forward(x);
    for (size_t i = 1U; i < layers.size(); ++i) {
      tmp = layers[i]->forward(tmp);
    }

    return tmp;
  }

  std::vector<ParameterPtr> parameters() override { return param_vector; }

  void save(std::ostream &) const override;
};
typedef std::shared_ptr<Sequential> SequentialPtr;
} // namespace Weed
