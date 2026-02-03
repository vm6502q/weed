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
 * GPU-accessible storage for complex data type elements
 */
struct GpuComplexStorage : public GpuStorage<complex> {
  GpuComplexStorage(const tcapint &n, const int64_t &did,
                    const bool &alloc = true)
      : GpuStorage<complex>(n, did, alloc) {}
  GpuComplexStorage(const std::vector<complex> &val, const int64_t &did)
      : GpuStorage<complex>(val, did) {}

  void FillZeros() override { dev->ClearRealBuffer(buffer, size << 1U); }
  void FillOnes() override { dev->FillOnesComplex(buffer, size); }
  void FillValue(const complex &v) override {
    dev->FillValueComplex(buffer, size, v);
  }

  complex operator[](const tcapint &idx) const override {
    if (idx >= GpuStorage<complex>::size) {
      throw std::invalid_argument(
          "GpuStorage::operator[] argument out-of-bounds!");
    }

    return dev->GetComplex(buffer, idx);
  }

  StoragePtr Upcast(const DType &dt) override { return get_ptr(); };

  StoragePtr cpu() override;
};
typedef std::shared_ptr<GpuComplexStorage> GpuComplexStoragePtr;
} // namespace Weed
