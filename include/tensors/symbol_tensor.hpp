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

#include "tensors/base_tensor.hpp"

namespace Weed {
/**
 * Non-mathematical tensor, solely for indexing (by integer enumeration)
 */
struct SymbolTensor : BaseTensor {
  SymbolTensor(const std::vector<tcapint> &shp,
               const std::vector<tcapint> &strd, const bool &rg = false,
               const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
               const int64_t &did = -1, const bool &s = true);
  SymbolTensor(const std::vector<tcapint> &val, const std::vector<tcapint> &shp,
               const std::vector<tcapint> &strd, const bool &rg = false,
               const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
               const int64_t &did = -1);
  SymbolTensor(const IntSparseVector &val, const std::vector<tcapint> &shp,
               const std::vector<tcapint> &strd, const bool &rg = false);
};
typedef std::shared_ptr<SymbolTensor> SymbolTensorPtr;
} // namespace Weed
