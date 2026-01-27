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

#include "storage/cpu_real_storage.hpp"
#include "storage/gpu_real_storage.hpp"

namespace Weed {
StoragePtr CpuRealStorage::gpu(int64_t did) {
  GpuRealStoragePtr cp = std::make_shared<GpuRealStorage>(size, did, false);
  cp->array = cp->Alloc(size);
  cp->AddAlloc(sizeof(real1) * size);
  cp->buffer = cp->MakeBuffer(size);

  return cp;
}
} // namespace Weed
