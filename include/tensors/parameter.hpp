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

#include "tensors/tensor.hpp"

namespace Weed {
struct Parameter;
typedef std::shared_ptr<Parameter> ParameterPtr;

/**
 * A Parameter is simply a tensor that always requires gradient calculation
 * during training and "lives" on a module"
 */
struct Parameter : Tensor {
  Parameter(const std::vector<tcapint> &shp, const std::vector<tcapint> &strd,
            const bool &s = true, const DType &dtype = DType::REAL,
            const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
            const int64_t &did = -1)
      : Tensor(shp, strd, true, s, DType::REAL, dtag, did) {}
  Parameter(const std::vector<real1> &val, const std::vector<tcapint> &shp,
            const std::vector<tcapint> &strd,
            const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
            const int64_t &did = -1)
      : Tensor(val, shp, strd, true, dtag, did) {}
  Parameter(const std::vector<complex> &val, const std::vector<tcapint> &shp,
            const std::vector<tcapint> &strd,
            const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
            const int64_t &did = -1)
      : Tensor(val, shp, strd, true, dtag, did) {}

  void train() { requires_grad = true; }
  void eval() { requires_grad = false; }

  void save(std::ostream &out) const;
  static ParameterPtr load(std::istream &in);
};
} // namespace Weed
