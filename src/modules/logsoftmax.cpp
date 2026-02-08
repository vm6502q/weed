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

#include "modules/logsoftmax.hpp"
#include "common/serializer.hpp"

namespace Weed {
TensorPtr LogSoftmax::forward(const TensorPtr x) {
  const tcapint ax = axis < 0 ? axis + x->shape.size() : axis;

  TensorPtr m = Tensor::max(x, ax);
  TensorPtr x_shifted = x - m;
  TensorPtr logsum = Tensor::log(Tensor::sum(Tensor::exp(x_shifted), ax));

  return x_shifted - logsum;
}
void LogSoftmax::save(std::ostream &os) const {
  Serializer::write_tcapint(os, axis);
}
} // namespace Weed
