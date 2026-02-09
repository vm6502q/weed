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

#include "modules/transformer_encoder_layer.hpp"
#include "common/serializer.hpp"

namespace Weed {
TensorPtr TransformerEncoderLayer::forward(const TensorPtr x) {
  // Self-attention block
  TensorPtr attn_out = self_attn->forward(x);
  TensorPtr x1 = norm1->forward(x + attn_out);

  // Feed-forward block
  TensorPtr ff = ff1->forward(x1);
  ff = activation->forward(ff);
  ff = ff2->forward(ff);

  TensorPtr out = norm2->forward(x1 + ff);

  return out;
}
void TransformerEncoderLayer::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, d_model);
  Serializer::write_tcapint(os, d_ff);
  Serializer::write_tcapint(os, num_heads);
  self_attn->save(os);
  ff1->save(os);
  ff2->save(os);
  norm1->save(os);
  norm2->save(os);
  activation->save(os);
}
} // namespace Weed
