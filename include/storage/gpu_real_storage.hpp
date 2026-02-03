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
 * GPU-accessible storage for real data type elements
 */
struct GpuRealStorage : public GpuStorage<real1> {
  GpuRealStorage(const tcapint &n, int64_t did, const bool &alloc = true)
      : GpuStorage<real1>(n, did, alloc) {}
  GpuRealStorage(const std::vector<real1> &val, const int64_t &did = -1)
      : GpuStorage<real1>(val, did) {}

  void FillZeros() override { dev->ClearRealBuffer(buffer, size); }
  void FillOnes() override { dev->FillOnesReal(buffer, size); }
  void FillValue(const real1 &v) override {
    dev->FillValueReal(buffer, size, v);
  }

  real1 operator[](const tcapint &idx) const override {
    if (idx >= GpuStorage<real1>::size) {
      throw std::invalid_argument(
          "GpuStorage::operator[] argument out-of-bounds!");
    }

    return dev->GetReal(buffer, idx);
  }

  StoragePtr Upcast(const DType &dt) override;

  StoragePtr cpu() override;
};
typedef std::shared_ptr<GpuRealStorage> GpuRealStoragePtr;
} // namespace Weed
