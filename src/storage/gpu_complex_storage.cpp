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

#include "storage/gpu_complex_storage.hpp"
#include "storage/cpu_complex_storage.hpp"

namespace Weed {
StoragePtr GpuComplexStorage::cpu() {
  CpuComplexStoragePtr cp = std::make_shared<CpuComplexStorage>(size);
  dev->LockSync(buffer, sizeof(complex) * size, cp->data.get(), false);

  return cp;
}
} // namespace Weed
