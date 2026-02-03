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

#include "storage/cpu_storage.hpp"

#include <vector>

namespace Weed {
/**
 * CPU-accessible storage for complex data type elements
 */
struct CpuComplexStorage : CpuStorage<complex> {
  CpuComplexStorage(const tcapint &n) : CpuStorage<complex>(n) {}
  CpuComplexStorage(const std::vector<complex> &i) : CpuStorage<complex>(i) {}
  StoragePtr Upcast(const DType &dt) override {
    return TypedStorage<complex>::get_ptr();
  }
  StoragePtr gpu(const int64_t &did = -1) override;
};
typedef std::shared_ptr<CpuComplexStorage> CpuComplexStoragePtr;
} // namespace Weed
