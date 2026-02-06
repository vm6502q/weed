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

#include "tensors/parameter.hpp"

namespace Weed {
void Parameter::save(std::ostream &out) const {}
ParameterPtr Parameter::load(std::istream &in, DeviceTag dtag_override) {
  return nullptr;
}
} // namespace Weed
