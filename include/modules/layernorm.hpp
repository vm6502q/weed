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
  LayerNorm(const tcapint &f, const real1 &e = FP_NORM_EPSILON,
            const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE)
      : Module(LAYERNORM_T), features(f), eps(e) {
    gamma = std::make_shared<Parameter>(std::vector<real1>(f, ZERO_R1),
                                        std::vector<tcapint>{f},
                                        std::vector<tcapint>{1}, dtag);
    gamma->storage->FillOnes();

    beta = std::make_shared<Parameter>(std::vector<real1>(f, ZERO_R1),
                                       std::vector<tcapint>{f},
                                       std::vector<tcapint>{1}, dtag);
    beta->storage->FillZeros();
  }

  TensorPtr forward(const TensorPtr x) override {
    // μ: (B, 1)
    TensorPtr mu = Tensor::mean(x, 1);

    // x − μ
    TensorPtr xc = x - mu;

    // σ²: (B, 1)
    TensorPtr var = Tensor::mean(xc * xc, 1);

    // normalized by sqrt(σ² + eps)
    TensorPtr y = xc / ((var + eps) ^ (ONE_R1 / 2));

    // affine transform
    y = y * gamma + beta;

    return y;
  }

  std::vector<ParameterPtr> parameters() override { return {gamma, beta}; }

  void save(std::ostream &) const override;
};
typedef std::shared_ptr<LayerNorm> LayerNormPtr;
} // namespace Weed
