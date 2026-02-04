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

#include <string>

namespace Weed {
/**
 * Validate that all tensors are on the same device, or throw otherwise
 */
void validate_all_same_device(const std::vector<const Tensor *> &t,
                              const std::string cls);
} // namespace Weed
