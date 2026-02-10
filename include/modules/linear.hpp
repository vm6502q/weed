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
 * Linear module to train $y = xW^T + b$
 */
struct Linear : public Module {
  tcapint in_features;
  tcapint out_features;

  ParameterPtr weight; // (in_features, out_features)
  ParameterPtr bias;   // (1, out_features) or null

  Linear() : Module(LINEAR_T){};
  Linear(tcapint in_f, tcapint out_f, bool use_bias = true,
         bool init_rand = true, DType dtype = DType::REAL,
         DeviceTag device = DeviceTag::DEFAULT_DEVICE, int64_t device_id = -1);

  TensorPtr forward(const TensorPtr x) override;
  std::vector<ParameterPtr> parameters() override;
  /**
   * Serialize storage to ostream
   */
  void save(std::ostream &) const override;
};
typedef std::shared_ptr<Linear> LinearPtr;
} // namespace Weed
