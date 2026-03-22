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

#include "modules/rope.hpp"
#include "common/serializer.hpp"
#include "ops/in_place.hpp"

// Rotary Position Embedding (contributed by Anthropic Claude)

namespace Weed {
void RoPE::_build_tables() {
  // Precompute theta_i = base^(-2i/head_dim) for i in [0, head_dim/2)
  // then for each position pos and each i:
  // angle = pos * theta_i
  // stored interleaved: [pos, 2i] = cos(angle), [pos, 2i+1] = cos(angle)
  // (same value for both elements of the pair — standard RoPE)

  const tcapint half = head_dim >> 1U;
  std::vector<real1> cos_data(max_seq_len * head_dim);
  std::vector<real1> sin_data(max_seq_len * head_dim);

  for (tcapint pos = 0U; pos < max_seq_len; ++pos) {
    for (tcapint i = 0U; i < half; ++i) {
      const real1_f theta =
          (real1_f)std::pow(base, -2.0f * (real1_f)i / (real1_f)head_dim);
      const real1_f angle = (real1_f)pos * theta;
      const real1 c = (real1)std::cos(angle);
      const real1 s = (real1)std::sin(angle);
      // Interleave: both elements of pair get same cos/sin
      // col-major: index = pos + (2i) * max_seq_len
      cos_data[pos + (2U * i) * max_seq_len] = c;
      cos_data[pos + (2U * i + 1U) * max_seq_len] = c;
      sin_data[pos + (2U * i) * max_seq_len] = s;
      sin_data[pos + (2U * i + 1U) * max_seq_len] = s;
    }
  }

  cos_table = std::make_shared<Tensor>(
      cos_data, std::vector<tcapint>{max_seq_len, head_dim}, false);
  sin_table = std::make_shared<Tensor>(
      sin_data, std::vector<tcapint>{max_seq_len, head_dim}, false);
}

TensorPtr RoPE::_rotate_half(const TensorPtr x) {
  // x shape: [B, H, T, head_dim]
  const symint B = (symint)x->shape[0U];
  const symint H = (symint)x->shape[1U];
  const symint T = (symint)x->shape[2U];
  const symint D = (symint)head_dim;
  const symint half = (symint)(head_dim >> 1U);

  // Allocate output container — same shape as x
  TensorPtr out = Tensor::zeros(
      {(tcapint)B, (tcapint)H, (tcapint)T, (tcapint)D}, x->requires_grad, false,
      x->storage->dtype, x->storage->device, x->storage->get_device_id());

  // Slice x into first and second halves along last dim
  TensorPtr x0 = Tensor::slice(x, 3, 0U, (tcapint)half);   // [..., :half]
  TensorPtr x1 = Tensor::slice(x, 3, half, (tcapint)half); // [..., half:]

  // Slice output container into two windows
  TensorPtr out0 = Tensor::slice(out, 3, 0U, (tcapint)half);   // [..., :half]
  TensorPtr out1 = Tensor::slice(out, 3, half, (tcapint)half); // [..., half:]

  // Fill: out = [-x1, x0]
  Weed::add_in_place(*out0, *(x1 * real1(-1.0f)));
  Weed::add_in_place(*out1, *x0);

  return out;
}

TensorPtr RoPE::forward(const TensorPtr x) {
  const symint T = (symint)x->shape[2U];

  TensorPtr c = Tensor::slice(cos_table, 0, 0U, (tcapint)T);
  TensorPtr s = Tensor::slice(sin_table, 0, 0U, (tcapint)T);

  // Broadcast cos/sin to [1, 1, T, head_dim]
  TensorPtr cos_b = Tensor::reshape(c, {1, 1, T, (symint)head_dim});
  TensorPtr sin_b = Tensor::reshape(s, {1, 1, T, (symint)head_dim});

  return x * cos_b + _rotate_half(x) * sin_b;
}

void RoPE::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, head_dim);
  Serializer::write_tcapint(os, max_seq_len);
  Serializer::write_real1_f(os, base);
  // cos/sin tables are recomputed on load — not saved
}
} // namespace Weed
