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

#include "storage/sparse_cpu_complex_storage.hpp"
#if ENABLE_GPU
#include "storage/gpu_complex_storage.hpp"
#endif

namespace Weed {
StoragePtr SparseCpuComplexStorage::gpu(const int64_t &did) {
#if ENABLE_GPU
  GpuComplexStoragePtr cp =
      std::make_shared<GpuComplexStorage>(size, did, false);
  cp->data = cp->Alloc(size);
  for (size_t i = 0U; i < size; ++i) {
    data[i] = (*this)[i];
  }
  cp->AddAlloc(sizeof(complex) * size);
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
