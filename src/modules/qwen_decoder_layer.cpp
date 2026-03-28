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

#include "modules/qwen_decoder_layer.hpp"
#include "common/serializer.hpp"

// "Swish-GLU" activation function (contributed by Anthropic Claude)

namespace Weed {
void QwenDecoderLayer::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, d_model);
  Serializer::write_tcapint(os, num_heads);
  Serializer::write_tcapint(os, num_kv_heads);
  self_attn->save(os);
  mlp->save(os);
  input_layernorm->save(os);
  post_attention_layernorm->save(os);
}
} // namespace Weed
