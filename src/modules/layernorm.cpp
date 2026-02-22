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

#include "modules/layernorm.hpp"
#include "common/serializer.hpp"

namespace Weed {
TensorPtr LayerNorm::forward(const TensorPtr x) {
  // μ: (B, 1)
  TensorPtr mu = Tensor::mean(x, axis);

  // x − μ
  TensorPtr xc = x - mu;

  // σ²: (B, 1)
  TensorPtr var = Tensor::mean(xc * xc, axis);

  // normalized by sqrt(σ² + eps)
  TensorPtr y = xc / ((var + eps) ^ real1(0.5f));

  // affine transform
  y = y * gamma + beta;

  return y;
}

void LayerNorm::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, features);
  Serializer::write_real(os, eps);
  gamma->save(os);
  beta->save(os);
}
} // namespace Weed
