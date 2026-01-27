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
 * Linear module to train $y = xW^T + b$
 */
struct Linear : public Module {
  ParameterPtr weight; // (out_features, in_features)
  ParameterPtr bias;   // (out_features) or null

  vecCapIntGpu in_features;
  vecCapIntGpu out_features;

  Linear(vecCapIntGpu in_f, vecCapIntGpu out_f, bool use_bias = true,
         DType dtype = DType::REAL,
         DeviceTag device = DeviceTag::DEFAULT_DEVICE, int64_t device_id = -1, bool init_rand = true);

  TensorPtr forward(const TensorPtr x) override;
  std::vector<ParameterPtr> parameters() override;
};
} // namespace Weed
