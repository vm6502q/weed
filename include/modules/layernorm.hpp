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

  ParameterPtr gamma; // scale
  ParameterPtr beta;  // shift

  LayerNorm() : Module(LAYERNORM_T) {}
  LayerNorm(const tcapint &f, const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
            const real1 &e = FP_NORM_EPSILON)
      : Module(LAYERNORM_T), features(f), eps(e) {
    gamma = std::make_shared<Parameter>(std::vector<real1>(f, real1(ZERO_R1)),
                                        std::vector<tcapint>{1U, 1U, f}, dtag);
    gamma->storage->FillOnes();

    beta = std::make_shared<Parameter>(std::vector<real1>(f, real1(ZERO_R1)),
                                       std::vector<tcapint>{1U, 1U, f}, dtag);
    beta->storage->FillZeros();
  }

  TensorPtr forward(const TensorPtr x) override;

  std::vector<ParameterPtr> parameters() override { return {gamma, beta}; }

  void save(std::ostream &) const override;
};
typedef std::shared_ptr<LayerNorm> LayerNormPtr;
} // namespace Weed
