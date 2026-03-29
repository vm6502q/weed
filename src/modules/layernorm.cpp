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
#include "modules/migrate_cpu.hpp"
#include "modules/migrate_gpu.hpp"
#include "common/serializer.hpp"

namespace Weed {
void LayerNorm::migrate_cpu() {
  MigrateCpuPtr mc = std::make_shared<MigrateCpu>();
  gamma = mc->pforward(gamma);
  beta = mc->pforward(beta);
}
void LayerNorm::migrate_gpu() {
  MigrateGpuPtr mg = std::make_shared<MigrateGpu>();
  gamma = mg->pforward(gamma);
  beta = mg->pforward(beta);
}

TensorPtr LayerNorm::forward(const TensorPtr x) {
  // x − μ
  TensorPtr xc = x - Tensor::mean(x, -1);

  // Mean-centered and normalized by sqrt(σ² + eps)
  TensorPtr y = xc / ((Tensor::mean(xc * xc, -1) + eps) ^ real1(0.5f));

  // affine transform
  y = y * gamma + beta;

  xc = nullptr;

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
