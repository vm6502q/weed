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
/**
 * Sum over a tensor index
 */
void reduce(const tcapint &index, const Tensor &a, Tensor &out);
void reduce_grad(const tcapint &index, Tensor &din, const Tensor &a,
                 const Tensor &dout);

/**
 * Maximum over a tensor index
 */
void max(const tcapint &index, const Tensor &a, Tensor &out);

/**
 * Minimum over a tensor index
 */
void min(const tcapint &index, const Tensor &a, Tensor &out);

void match_grad(const tcapint &index, Tensor &din, const Tensor &a,
                const Tensor &dout, const Tensor &out);
} // namespace Weed
