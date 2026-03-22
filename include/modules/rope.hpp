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

#pragma once

#include "modules/module.hpp"

namespace Weed {
/**
 * Rotary Position Embedding (contributed by Anthropic Claude)
 */
struct RoPE : public Module {
  tcapint head_dim;
  tcapint max_seq_len;
  real1_f base;
  TensorPtr cos_table; // [max_seq_len, head_dim]
  TensorPtr sin_table; // [max_seq_len, head_dim]

  RoPE() : Module(ROPE_T) {}
  RoPE(const tcapint &head_dim_, const tcapint &max_seq_len_ = 2048U,
       const real1_f &base_ = 10000.0f)
      : Module(ROPE_T), head_dim(head_dim_), max_seq_len(max_seq_len_),
        base(base_) {
    _build_tables();
  }

  void _build_tables();
  TensorPtr _rotate_half(const TensorPtr x);
  TensorPtr forward(const TensorPtr x) override;
  void save(std::ostream &os) const override;
};
typedef std::shared_ptr<RoPE> RoPEPtr;
} // namespace Weed
