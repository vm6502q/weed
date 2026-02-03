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

#include "storage/sparse_cpu_storage.hpp"

namespace Weed {
/**
 * CPU-accessible storage for real data type elements
 */
struct SparseCpuComplexStorage : SparseCpuStorage<complex> {
  SparseCpuComplexStorage(const ComplexSparseVector &v, const tcapint &n)
      : SparseCpuStorage<complex>(v, n) {}
  SparseCpuComplexStorage(const tcapint &n) : SparseCpuStorage<complex>(n) {}
  StoragePtr Upcast(const DType &dt) override {
    return SparseCpuStorage<complex>::get_ptr();
  }
  StoragePtr gpu(const int64_t &did = -1) override;
};
typedef std::shared_ptr<SparseCpuComplexStorage> SparseCpuComplexStoragePtr;
} // namespace Weed
