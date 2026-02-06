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
#include "common/serializer.hpp"
#include "storage/cpu_real_storage.hpp"
#include "storage/gpu_complex_storage.hpp"

namespace Weed {
StoragePtr GpuRealStorage::cpu() {
  CpuRealStoragePtr cp = std::make_shared<CpuRealStorage>(size);
  dev->LockSync(buffer, sizeof(real1) * size, cp->data.get(), false);

  return cp;
}
StoragePtr GpuRealStorage::Upcast(const DType &dt) {
  if (dt != DType::COMPLEX) {
    return get_ptr();
  }

  GpuComplexStoragePtr n =
      std::make_shared<GpuComplexStorage>(size, dev->deviceID);
  dev->UpcastRealBuffer(buffer, n->buffer, size);

  return n;
}
void GpuRealStorage::save(std::ostream &os) const {
  Storage::save(os);
  if (data) {
    dev->LockSync(buffer, sizeof(real1) * size, data.get(), false);
    for (tcapint i = 0U; i < size; ++i) {
      Serializer::write_real(os, data.get()[i]);
    }
    dev->UnlockSync(buffer, data.get());
  } else {
    std::unique_ptr<real1[], void (*)(real1 *)> d(Alloc(size));
    dev->LockSync(buffer, sizeof(real1) * size, d.get(), false);
    for (tcapint i = 0U; i < size; ++i) {
      Serializer::write_real(os, d.get()[i]);
    }
  }
}
} // namespace Weed
