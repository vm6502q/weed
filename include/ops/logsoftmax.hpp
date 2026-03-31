//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of
// Qrack.
//
// This file was produced by (Anthropic) Claude based on softmax.hpp.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#pragma once

#include "tensors/tensor.hpp"

namespace Weed {
/**
 * Fused log-softmax forward:
 *   out[i] = (a[i] - max(a)) - log(sum(exp(a - max(a))))
 * along the given axis. Numerically stable, single-pass allocation.
 */
void logsoftmax(const tcapint &index, const Tensor &a, Tensor &out);

/**
 * Fused log-softmax backward:
 *   din += dout - exp(out) * sum(dout, axis)
 * Requires the log-softmax output (out) from the forward pass.
 */
void logsoftmax_grad(const tcapint &index, Tensor &din, const Tensor &out,
                     const Tensor &dout);
} // namespace Weed
