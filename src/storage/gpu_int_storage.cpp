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

#include "storage/gpu_int_storage.hpp"
#include "common/serializer.hpp"
#include "storage/cpu_int_storage.hpp"

namespace Weed {
StoragePtr GpuIntStorage::cpu() {
  CpuIntStoragePtr cp = std::make_shared<CpuIntStorage>(size);
  dev->LockSync(buffer, sizeof(tcapint) * size, cp->data.get(), false);

  return cp;
}
void GpuIntStorage::save(std::ostream &os) const {
  Storage::save(os);
  if (data) {
    dev->LockSync(buffer, sizeof(symint) * size, data.get(), false);
    for (tcapint i = 0U; i < size; ++i) {
      Serializer::write_symint(os, data.get()[i]);
    }
    dev->UnlockSync(buffer, data.get());
  } else {
    std::unique_ptr<symint[], void (*)(symint *)> d(Alloc(size));
    dev->LockSync(buffer, sizeof(symint) * size, d.get(), false);
    for (tcapint i = 0U; i < size; ++i) {
      Serializer::write_symint(os, d.get()[i]);
    }
  }
}
} // namespace Weed
