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
 * Root-mean-square norm
 */
struct RMSNorm : public Module {
  symint axis;
  tcapint hidden_size;
  ParameterPtr weight;

  RMSNorm() : Module(RMS_NORM_T) {}
  RMSNorm(const tcapint &hidden_size_, const symint &axis_ = -1)
      : Module(RMS_NORM_T), axis(axis_), hidden_size(hidden_size_) {
    weight = std::make_shared<Parameter>(std::vector<tcapint>{hidden_size},
                                         std::vector<tcapint>{1U}, false);
    weight->storage->FillOnes();
  }
  std::vector<ParameterPtr> parameters() override { return {weight}; }
  TensorPtr forward(const TensorPtr x) override {
    return (x / ((Tensor::mean(x * x, axis) + real1_f(FP_NORM_EPSILON)) ^
                 real1_f(0.5f))) *
           weight;
  }
  void save(std::ostream &os) const override {
    Module::save(os);
    Serializer::write_symint(os, axis);
    Serializer::write_tcapint(os, hidden_size);
    weight->save(os);
  }
};
typedef std::shared_ptr<RMSNorm> RMSNormPtr;
} // namespace Weed
