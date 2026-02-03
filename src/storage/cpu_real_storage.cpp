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
#include "storage/cpu_complex_storage.hpp"
#if ENABLE_GPU
#include "storage/gpu_real_storage.hpp"
#endif

namespace Weed {
StoragePtr CpuRealStorage::gpu(const int64_t &did) {
#if ENABLE_GPU
  GpuRealStoragePtr cp = std::make_shared<GpuRealStorage>(size, did, false);
  cp->data = cp->Alloc(size);
  std::copy(data.get(), data.get() + size, cp->data.get());
  cp->AddAlloc(sizeof(real1) * size);
  cp->buffer = cp->MakeBuffer(size);
  if (!(cp->dev->device_context->use_host_mem)) {
    cp->data = nullptr;
  }

  return cp;
#else
  return get_ptr();
#endif
}
StoragePtr CpuRealStorage::Upcast(const DType &dt) {
  if (dt == DType::REAL) {
    return get_ptr();
  }

  CpuComplexStoragePtr n = std::make_shared<CpuComplexStorage>(size);
  std::transform(data.get(), data.get() + size, n->data.get(),
                 [](real1 v) { return complex(v, ZERO_R1); });

  return n;
}
} // namespace Weed
