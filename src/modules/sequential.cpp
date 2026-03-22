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

#include "modules/sequential.hpp"
#include "common/serializer.hpp"

namespace Weed {
void Sequential::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, (tcapint)(layers.size()));
  for (size_t i = 0U; i < layers.size(); ++i) {
    layers[i]->save(os);
  }
}
} // namespace Weed
