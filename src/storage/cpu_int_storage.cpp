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

#include "storage/cpu_int_storage.hpp"
#include "common/serializer.hpp"
#if ENABLE_GPU
#include "storage/gpu_int_storage.hpp"
#endif

namespace Weed {
StoragePtr CpuIntStorage::gpu(const int64_t &did) {
#if ENABLE_GPU
  GpuIntStoragePtr cp = std::make_shared<GpuIntStorage>(size, did, false);
  cp->data = cp->Alloc(size);
  std::copy(data.get(), data.get() + size, cp->data.get());
  cp->buffer = cp->MakeBuffer(size);
  if (!(cp->dev->device_context->use_host_mem)) {
    cp->data = nullptr;
  }

  return cp;
#else
  return get_ptr();
#endif
}
void CpuIntStorage::save(std::ostream &os) const {
  Storage::save(os);
  for (tcapint i = 0U; i < size; ++i) {
    Serializer::write_symint(os, data[i]);
  }
}
} // namespace Weed
