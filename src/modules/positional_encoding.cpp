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

#include "modules/positional_encoding.hpp"
#include "common/serializer.hpp"
#include "modules/migrate_cpu.hpp"
#include "modules/migrate_gpu.hpp"

namespace Weed {
void PositionalEncoding::migrate_cpu() {
  MigrateCpuPtr mc = std::make_shared<MigrateCpu>();
  pe = mc->pforward(pe);
}
void PositionalEncoding::migrate_gpu() {
  MigrateGpuPtr mg = std::make_shared<MigrateGpu>();
  pe = mg->pforward(pe);
}
PositionalEncoding::PositionalEncoding(tcapint max_seq_len_, tcapint d_model_,
                                       real1_f pos_val_, DeviceTag device)
    : Module(POSITIONAL_ENCODING_T), max_seq_len(max_seq_len_),
      d_model(d_model_), pos_val(pos_val_) {

  // pe has shape [max_seq_len, d_model], and tensors are column-major, so
  // element (pos, col) is at index pos + max_seq_len * col.
  const tcapint max_i = d_model >> 1U;
  std::vector<real1> values(max_seq_len * d_model);
  for (tcapint pos = 0; pos < max_seq_len; ++pos) {
    for (tcapint i = 0; i < max_i; ++i) {
      const tcapint col = i << 1U;
      const real1 coeff =
          (real1)(1.0 / std::pow(pos_val, ((real1)(i << 1U)) / d_model));

      values[pos + max_seq_len * col] = std::cos(coeff * pos);
      values[pos + max_seq_len * (col + 1U)] = std::sin(coeff * pos);
    }
    if (d_model & 1U) {
      const tcapint col = max_i << 1U;
      const real1 div = (real1)std::pow(pos_val, (2.0 * max_i) / d_model);
      values[pos + max_seq_len * col] = std::cos(pos / div);
    }
  }

  pe = std::make_shared<Parameter>(
      values, std::vector<tcapint>{max_seq_len, d_model}, device);

  // Never requires_grad
  pe->eval();
}
TensorPtr PositionalEncoding::forward(const TensorPtr x) {
  // x: [B, T, D]
  const tcapint T = x->shape[1U];

  if ((seq_offset + T) > max_seq_len) {
    throw std::invalid_argument(
        "Input sequence length exceeds maximum positional encoding length!");
  }

  // slice pe -> [T, D] (starting after any tokens already in the KV cache)
  // broadcast pe_slice to [B, T, D] and add
  return x + Tensor::slice(pe, 0, seq_offset, T);
}
void PositionalEncoding::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, max_seq_len);
  Serializer::write_tcapint(os, d_model);
  Serializer::write_real1_f(os, pos_val);
}
} // namespace Weed
