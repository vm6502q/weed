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

#include "modules/softmax.hpp"
#include "common/serializer.hpp"

namespace Weed {
TensorPtr Softmax::forward(const TensorPtr x) {
  // Normalize axis
  tcapint ax = axis;
  if (ax < 0) {
    ax += x->shape.size();
  }

  // m = max(x, axis)
  TensorPtr m = Tensor::max(x); //, ax);

  // x_shifted = x - m
  TensorPtr x_shifted = x - m;

  // ex = exp(x_shifted)
  TensorPtr ex = Tensor::exp(x_shifted);

  // denom = sum(ex, axis)
  TensorPtr denom = Tensor::sum(ex, ax);

  // softmax = ex / denom
  return ex / denom;
}
void Softmax::save(std::ostream &os) const {
  Serializer::write_tcapint(os, axis);
  // Needs the inheriting struct to do the rest
}
} // namespace Weed
