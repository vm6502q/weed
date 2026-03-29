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

#include "modules/linear.hpp"

namespace Weed {
/**
 * "Swish-GLU" activation function (contributed by Anthropic Claude)
 */
struct SwiGLU : public Module {
  tcapint hidden_size;
  tcapint intermediate_size;
  LinearPtr gate_proj;
  LinearPtr up_proj;
  LinearPtr down_proj;

  std::vector<ParameterPtr> param_vector;

  SwiGLU() : Module(SWIGLU_T) {}
  SwiGLU(const tcapint &hidden_size_, const tcapint &intermediate_size_)
      : Module(SWIGLU_T), hidden_size(hidden_size_),
        intermediate_size(intermediate_size_) {
    gate_proj = std::make_shared<Linear>(hidden_size, intermediate_size, false);
    up_proj = std::make_shared<Linear>(hidden_size, intermediate_size, false);
    down_proj = std::make_shared<Linear>(intermediate_size, hidden_size, false);
  }

  void _register_params() {
    auto g = gate_proj->parameters();
    auto u = up_proj->parameters();
    auto d = down_proj->parameters();
    param_vector.insert(param_vector.end(), g.begin(), g.end());
    param_vector.insert(param_vector.end(), u.begin(), u.end());
    param_vector.insert(param_vector.end(), d.begin(), d.end());
  }

  std::vector<ParameterPtr> parameters() override { return param_vector; }

  void train() override {
    gate_proj->train();
    up_proj->train();
    down_proj->train();
  }
  void eval() override {
    gate_proj->eval();
    up_proj->eval();
    down_proj->eval();
  }

  void migrate_cpu() override {
    gate_proj->migrate_cpu();
    up_proj->migrate_cpu();
    down_proj->migrate_cpu();
  }
  void migrate_gpu() override {
    gate_proj->migrate_gpu();
    up_proj->migrate_gpu();
    down_proj->migrate_gpu();
  }

  TensorPtr forward(const TensorPtr x) override {
    TensorPtr gate = gate_proj->forward(x);
    TensorPtr up = up_proj->forward(x);
    // SiLU(gate) * up
    TensorPtr activated = gate * Tensor::sigmoid(gate) * up;
    return down_proj->forward(activated);
  }

  void save(std::ostream &os) const override;
};
typedef std::shared_ptr<SwiGLU> SwiGLUPtr;
} // namespace Weed
