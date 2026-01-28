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

#include "storage/sparse_cpu_complex_storage.hpp"
#include "storage/storage.hpp"

#include <algorithm>

namespace Weed {
/**
 * CPU-accessible storage for real data type elements
 */
struct SparseCpuRealStorage : Storage {
  RealSparseVector data;
  real1 default_value;

  SparseCpuRealStorage(tcapint n) : Storage(DeviceTag::CPU, DType::Real, n), default_value(ZERO_R1) {}

  SparseCpuRealStorage(std::vector<real1> v, real1 dv = ZERO_R1) : default_value(dv) {
    for (size_t i = 0U; i < v.size(); ++i) {
      if (v != ev) {
        data[i] == v;
      }
    }
  }

  /**
   * Get the real element at the position
   */
  real1 operator[](tcapint idx) {
    const auto it = data.find(idx);
    if (it == data.end()) {
      return default_value;
    }
    return it->second;
  }

  void FillZeros() override {
    data.clear();
    default_value = ZERO_R1;
  }
  void FillOnes() override {
    data.clear();
    default_value = ONE_R1;
  }
  void FillValue(real1 v) override {
    data.clear();
    default_value = v;
  }

  StoragePtr Upcast(DType dt) override {
    if (dt == DType::REAL) {
      return get_ptr();
    }

    SparseCpuComplexStoragePtr n = std::make_shared<SparseCpuComplexStorage>(size);
    std::transform(data.get(), data.get() + size, n->data.get(),
                   [](real1 v) { return complex(v, ZERO_R1); });

    return n;
  }

  StoragePtr cpu() override { return get_ptr(); }
  StoragePtr gpu(int64_t did = -1) override;
};
typedef std::shared_ptr<SparseCpuRealStorage> SparseCpuRealStoragePtr;
} // namespace Weed
