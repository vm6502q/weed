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
 * Learned positional encoding.
 *
 * Adds a trainable (1, max_len, d_model) tensor
 * to input of shape (B, T, d_model).
 */
struct LearnedPositionalEncoding : public Module {
  tcapint max_len;
  tcapint d_model;

  ParameterPtr pos_encoding; // (1, max_len, d_model)

  LearnedPositionalEncoding() : Module(LEARNED_POSITIONAL_ENCODING_T) {}

  LearnedPositionalEncoding(const tcapint &max_len_, const tcapint &d_model_,
                            const DeviceTag &dtag = DEFAULT_DEVICE);

  TensorPtr forward(const TensorPtr x) override;

  std::vector<ParameterPtr> parameters() override { return {pos_encoding}; }

  void save(std::ostream &) const override;
};

typedef std::shared_ptr<LearnedPositionalEncoding> LearnedPositionalEncodingPtr;

} // namespace Weed
