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

#include "modules/dropout.hpp"
#include "common/serializer.hpp"

#include <random>

namespace Weed {
TensorPtr Dropout::forward(const TensorPtr x) {
  if (!training || p == ZERO_R1) {
    return x;
  }

  std::uniform_real_distribution<real1_s> dis((real1_s)ZERO_R1,
                                              (real1_s)ONE_R1);
  std::random_device rd;
  std::mt19937 gen(rd());

  // Fill mask with Bernoulli(1 - p)
  // mask[i] = 1 with prob (1 - p), else 0
  const tcapint sz = x->get_broadcast_size();
  RealSparseVector m;
  for (tcapint n = 0; n < sz; ++n) {
    if (dis(gen) > p) {
      m[n] = ONE_R1;
    }
  }

  // Allocate mask like x (real-valued)
  mask = std::make_shared<Tensor>(m, x->shape, x->stride);

  // y = x * mask / (1 - p)
  return (x * mask) / real1(ONE_R1 - p);
}
void Dropout::save(std::ostream &os) const {
  Serializer::write_real(os, p);
  Serializer::write_bool(os, training);
}
} // namespace Weed
