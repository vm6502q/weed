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
#include "common/serializer.hpp"
#include "storage/cpu_complex_storage.hpp"

namespace Weed {
StoragePtr GpuComplexStorage::cpu() {
  CpuComplexStoragePtr cp = std::make_shared<CpuComplexStorage>(size);
  dev->LockSync(buffer, sizeof(complex) * size, cp->data.get(), false);

  return cp;
}
void GpuComplexStorage::save(std::ostream &os) const {
  Storage::save(os);
  if (data) {
    dev->LockSync(buffer, sizeof(complex) * size, data.get(), false);
    for (tcapint i = 0U; i < size; ++i) {
      Serializer::write_complex(os, data.get()[i]);
    }
    dev->UnlockSync(buffer, data.get());
  } else {
    std::unique_ptr<complex[], void (*)(complex *)> d(Alloc(size));
    dev->LockSync(buffer, sizeof(complex) * size, d.get(), false);
    for (tcapint i = 0U; i < size; ++i) {
      Serializer::write_complex(os, d.get()[i]);
    }
  }
}
} // namespace Weed
