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

  // [max_seq_len, d_model], requires_grad=false
  ParameterPtr pe;

  PositionalEncoding(tcapint max_seq_len, tcapint d_model,
                     DeviceTag device = DEFAULT_DEVICE);

  TensorPtr forward(const TensorPtr x) override;

  void save(std::ostream &) const override;
};
typedef std::shared_ptr<PositionalEncoding> PositionalEncodingPtr;
} // namespace Weed
