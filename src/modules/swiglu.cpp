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

#include "modules/swiglu.hpp"
#include "common/serializer.hpp"

// "Swish-GLU" activation function (contributed by Anthropic Claude)

namespace Weed {
void SwiGLU::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, hidden_size);
  Serializer::write_tcapint(os, intermediate_size);
  gate_proj->save(os);
  up_proj->save(os);
  down_proj->save(os);
}
} // namespace Weed
