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
 * CPU-accessible storage for integer-value data type elements
 */
struct CpuIntStorage : CpuStorage<symint> {
  CpuIntStorage(const tcapint &n) : CpuStorage<symint>(n) {}
  CpuIntStorage(const std::vector<symint> &i) : CpuStorage<symint>(i) {}
  StoragePtr Upcast(const DType &dt) override {
    throw std::domain_error("Don't up-cast integer type (for symbol tables)!");
  }
  StoragePtr gpu(const int64_t &did = -1) override;
};
typedef std::shared_ptr<CpuIntStorage> CpuIntStoragePtr;
} // namespace Weed
