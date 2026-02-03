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

#include "storage/cpu_complex_storage.hpp"
#if ENABLE_GPU
#include "storage/gpu_complex_storage.hpp"
#endif

namespace Weed {
StoragePtr CpuComplexStorage::gpu(const int64_t &did) {
#if ENABLE_GPU
  GpuComplexStoragePtr cp =
      std::make_shared<GpuComplexStorage>(size, did, false);
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
} // namespace Weed
