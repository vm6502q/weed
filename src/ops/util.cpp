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

#include "ops/util.hpp"

namespace Weed {
/**
 * Validate that all tensors are on the same device, or throw otherwise
 */
void validate_all_same_device(const std::vector<const Tensor *> &t,
                              const std::string cls) {
#if ENABLE_GPU
  if (t.size() < 2U) {
    return;
  }

  const DeviceTag dtag = t[0U]->storage->device;
  for (const auto &x : t) {
    if (dtag != x->storage->device) {
      throw std::domain_error(
          std::string("In ") + cls +
          std::string(", tensor storage devices do not match!"));
    }
  }
#endif
}
} // namespace Weed
