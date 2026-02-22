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
 * Learn normalization across features, per match row
 */
struct LayerNorm : Module {
  tcapint features;
  real1 eps;
  symint axis;

  ParameterPtr gamma; // scale
  ParameterPtr beta;  // shift

  LayerNorm() : Module(LAYERNORM_T) {}
  LayerNorm(const tcapint &f, const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
            const real1 &e = FP_NORM_EPSILON, const symint &a = -1)
      : Module(LAYERNORM_T), features(f), eps(e), axis(a) {
    gamma = std::make_shared<Parameter>(std::vector<real1>(f, real1(ZERO_R1)),
                                        std::vector<tcapint>{f},
                                        std::vector<tcapint>{1}, dtag);
    gamma->storage->FillOnes();

    beta = std::make_shared<Parameter>(std::vector<real1>(f, real1(ZERO_R1)),
                                       std::vector<tcapint>{f},
                                       std::vector<tcapint>{1}, dtag);
    beta->storage->FillZeros();
  }

  TensorPtr forward(const TensorPtr x) override;

  std::vector<ParameterPtr> parameters() override { return {gamma, beta}; }

  void save(std::ostream &) const override;
};
typedef std::shared_ptr<LayerNorm> LayerNormPtr;
} // namespace Weed
