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

#include "tensors/symbol_tensor.hpp"
#include "tensors/tensor.hpp"

namespace Weed {
/**
 * Embedding forward function
 */
void embedding_gather(const SymbolTensor &, const Tensor &, Tensor &);
/**
 * Embedding backward function
 */
void embedding_scatter_add(Tensor &a, const SymbolTensor &, const Tensor &);
} // namespace Weed
