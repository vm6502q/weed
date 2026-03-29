//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of
// Qrack.
//
// This file was produced by (Anthropic) Claude based on reduce.hpp.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#pragma once

#include "tensors/tensor.hpp"

namespace Weed {
/**
 * Fused softmax forward: out[i] = exp(a[i] - max(a)) / sum(exp(a - max(a)))
 * along the given axis. Numerically stable, single-pass allocation.
 */
void softmax(const tcapint &index, const Tensor &a, Tensor &out);

/**
 * Fused softmax backward: din += out * (dout - sum(dout * out, axis))
 * Requires the softmax output (out) from the forward pass.
 */
void softmax_grad(const tcapint &index, Tensor &din, const Tensor &out,
                  const Tensor &dout);
} // namespace Weed
