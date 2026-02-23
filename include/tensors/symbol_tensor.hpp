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
struct SymbolTensor;
typedef std::shared_ptr<SymbolTensor> SymbolTensorPtr;

/**
 * Non-mathematical tensor, solely for indexing (by integer enumeration)
 */
struct SymbolTensor : BaseTensor {
  SymbolTensor(const std::vector<tcapint> &shp,
               const std::vector<tcapint> &strd, const bool &rg = false,
               const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
               const int64_t &did = -1, const bool &s = true);
  SymbolTensor(const std::vector<symint> &val, const std::vector<tcapint> &shp,
               const bool &rg = false,
               const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
               const int64_t &did = -1);

  using BaseTensor::reshape;
  /**
   * Reshape the tensor
   */
  static SymbolTensorPtr reshape(const SymbolTensorPtr a,
                                 const std::vector<symint> &s) {
    SymbolTensorPtr out = std::make_shared<SymbolTensor>(*(a.get()));
    out->reshape(s);

    return out;
  }

  using BaseTensor::transpose;
  /**
   * If the tensor has exactly two indices, transpose them
   */
  static SymbolTensorPtr transpose(const SymbolTensorPtr a) {
    SymbolTensorPtr out = std::make_shared<SymbolTensor>(*(a.get()));
    out->transpose();

    return out;
  }

  /**
   * Transpose the two tensor indices
   */
  static SymbolTensorPtr transpose(const SymbolTensorPtr a, symint i,
                                   symint j) {
    SymbolTensorPtr out = std::make_shared<SymbolTensor>(*(a.get()));
    out->transpose(i, j);

    return out;
  }
};
} // namespace Weed
