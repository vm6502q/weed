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

#include "storage/gpu_real_storage.hpp"
#include "storage/cpu_real_storage.hpp"

namespace Weed {
StoragePtr GpuRealStorage::cpu() {
  CpuRealStoragePtr cp = std::make_shared<CpuRealStorage>(size);
  is_mapped =
      dev->LockSync(buffer, sizeof(real1) * size, cp->data.get(), false);

  return cp;
}
} // namespace Weed
