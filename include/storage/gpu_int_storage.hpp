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

#pragma once

#include "storage/gpu_storage.hpp"

#if !ENABLE_OPENCL && !ENABLE_CUDA
#error GPU files were included without either OpenCL and CUDA enabled.
#endif

namespace Weed {
/**
 * GPU-accessible storage for integer-value data type elements
 */
struct GpuIntStorage : public GpuStorage<symint> {
  GpuIntStorage(const tcapint &n, const int64_t &did, const bool &alloc = true)
      : GpuStorage<symint>(INT_GPU_DENSE, n, did, alloc) {}
  GpuIntStorage(const std::vector<symint> &val, const int64_t &did = -1)
      : GpuStorage<symint>(INT_GPU_DENSE, val, did) {}

  void FillZeros() override { dev->ClearIntBuffer(buffer, size); }
  void FillOnes() override { dev->FillOnesInt(buffer, size); }
  void FillValue(const symint &v) override {
    dev->FillValueInt(buffer, size, v);
  }

  // TODO:
  symint operator[](const tcapint &idx) const override {
    if (idx >= GpuStorage<symint>::size) {
      throw std::invalid_argument(
          "GpuStorage::operator[] argument out-of-bounds!");
    }

    return dev->GetInt(buffer, idx);
  }

  StoragePtr Upcast(const DType &dt) override {
    throw std::domain_error("Don't up-cast integer type (for symbol tables)!");
  }

  StoragePtr cpu() override;

  void save(std::ostream &) const override;
};
typedef std::shared_ptr<GpuIntStorage> GpuIntStoragePtr;
} // namespace Weed
