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
struct SparseCpuRealStorage : SparseCpuStorage<real1> {
  SparseCpuRealStorage(const RealSparseVector &v, const tcapint &n)
      : SparseCpuStorage<real1>(REAL_CPU_SPARSE, v, n) {}
  SparseCpuRealStorage(const tcapint &n)
      : SparseCpuStorage<real1>(REAL_CPU_SPARSE, n) {}
  StoragePtr Upcast(const DType &dt) override;
  StoragePtr gpu(const int64_t &did = -1) override;
  void save(std::ostream &) const override;
};
typedef std::shared_ptr<SparseCpuRealStorage> SparseCpuRealStoragePtr;
} // namespace Weed
