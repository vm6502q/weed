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

#include "common/weed_types.hpp"

#include <vector>

namespace Weed {
/**
 * Non-mathematical tensor, solely for indexing (by integer enumeration)
 */
struct SymbolTensor {
  std::vector<tcapint> data;
  std::vector<tcapint> shape;
  std::vector<tcapint> stride;
};
typedef std::shared_ptr<SymbolTensor> SymbolTensorPtr;
} // namespace Weed
