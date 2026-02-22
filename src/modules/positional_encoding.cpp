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

namespace Weed {
PositionalEncoding::PositionalEncoding(tcapint max_seq_len_, tcapint d_model_,
                                       DeviceTag device, const symint& a)
    : Module(POSITIONAL_ENCODING_T), max_seq_len(max_seq_len_),
      d_model(d_model_), axis(a) {

  const tcapint max_i = d_model >> 1U;
  std::vector<real1> values(max_seq_len * d_model);
  for (tcapint pos = 0; pos < max_seq_len; ++pos) {
    for (tcapint i = 0; i < max_i; ++i) {
      const tcapint idx = pos * d_model + (i << 1U);
      const real1 coeff =
          (real1)(1.0 / std::pow(8192.0, ((real1)(i << 1U)) / d_model));

      values[idx] = std::cos(coeff * pos);
      values[idx + 1U] = std::sin(coeff * pos);
    }
    if (d_model & 1U) {
      const tcapint idx = pos * d_model + (max_i << 1U);
      const real1 div = (real1)std::pow(8192.0, (2.0 * max_i) / d_model);
      values[idx] = std::cos(pos / div);
    }
  }

  pe = std::make_shared<Parameter>(
      values, std::vector<tcapint>{max_seq_len, d_model},
      Tensor::full_contiguous_stride({max_seq_len, d_model}), device);

  // Never requires_grad
  pe->eval();
}
TensorPtr PositionalEncoding::forward(const TensorPtr x) {
  symint ap1 = axis + 1;
  while (ap1 >= (symint)(x->shape.size())) {
    ap1 -= x->shape.size();
  }
  while (ap1 < 0) {
     ap1 += x->shape.size();
  }
  symint a = axis;
  while (a < 0) {
     a += x->shape.size();
  }

  // x: [B, T, D]
  const tcapint T = x->shape[a + 1U];

  // slice pe -> [T, D]
  TensorPtr pe_slice = Tensor::slice(pe, a, 0, T);

  // broadcast pe_slice to [B, T, D] and add
  return x + pe_slice;
}
void PositionalEncoding::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, max_seq_len);
  Serializer::write_tcapint(os, d_model);
}
} // namespace Weed
