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

#include "storage/sparse_cpu_real_storage.hpp"
#include "storage/gpu_real_storage.hpp"

namespace Weed {
StoragePtr SparseCpuRealStorage::gpu(const int64_t &did) {
  GpuRealStoragePtr cp = std::make_shared<GpuRealStorage>(size, did, false);
  cp->array = cp->Alloc(size);
  for (size_t i = 0U; i < size; ++i) {
    data[i] = (*this)[i];
  }
  cp->AddAlloc(sizeof(real1) * size);
  cp->buffer = cp->MakeBuffer(size);
  if (!(cp->dev->device_context->use_host_mem)) {
    cp->array.reset();
  }

  return cp;
}

StoragePtr SparseCpuRealStorage::Upcast(const DType &dt) {
  if (dt == DType::REAL) {
    return get_ptr();
  }

  SparseCpuComplexStoragePtr n =
      std::make_shared<SparseCpuComplexStorage>(size);
  for (auto it = data.begin(); it != data.end(); ++it) {
    n->data[it->first] = (complex)it->second;
  }

  return n;
}
} // namespace Weed
