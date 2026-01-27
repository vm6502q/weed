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

#include "tensor.hpp"

namespace Weed {
/**
 * A Parameter is simply a tensor that always requires gradient calculation and
 * "lives" on a module"
 */
struct Parameter : Tensor {
  Parameter(std::vector<tcapint> shp, std::vector<tcapint> strd,
            DType dtype = DType::REAL,
            DeviceTag dtag = DeviceTag::DEFAULT_DEVICE, int64_t did = -1)
      : Tensor(shp, strd, true, DType::REAL, dtag, did) {}
  Parameter(std::vector<real1> val, std::vector<tcapint> shp,
            std::vector<tcapint> strd,
            DeviceTag dtag = DeviceTag::DEFAULT_DEVICE, int64_t did = -1)
      : Tensor(val, shp, strd, true, dtag, did) {}
  Parameter(std::vector<complex> val, std::vector<tcapint> shp,
            std::vector<tcapint> strd,
            DeviceTag dtag = DeviceTag::DEFAULT_DEVICE, int64_t did = -1)
      : Tensor(val, shp, strd, true, dtag, did) {}
};
typedef std::shared_ptr<Parameter> ParameterPtr;
} // namespace Weed
