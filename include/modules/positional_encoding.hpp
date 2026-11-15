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
#include "tensors/tensor.hpp"

namespace Weed {
struct PositionalEncoding : public Module {
  tcapint max_seq_len;
  tcapint d_model;
  real1_f pos_val;

  // [max_seq_len, d_model], requires_grad=false
  ParameterPtr pe;

  // Position of the first input token (nonzero only when continuing from a KV
  // cache; runtime state, not serialized)
  tcapint seq_offset = 0U;

  PositionalEncoding(tcapint max_seq_len, tcapint d_model,
                     real1_f pos_val_ = 8192.0,
                     DeviceTag device = DEFAULT_DEVICE);

  void migrate_cpu() override;
  void migrate_gpu() override;

  void set_seq_offset(tcapint o) override { seq_offset = o; }
  void reset_cache() override { seq_offset = 0U; }

  TensorPtr forward(const TensorPtr x) override;

  void save(std::ostream &) const override;
};
typedef std::shared_ptr<PositionalEncoding> PositionalEncodingPtr;
} // namespace Weed
