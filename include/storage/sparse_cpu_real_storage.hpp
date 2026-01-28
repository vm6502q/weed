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
struct SparseCpuRealStorage : RealStorage {
  RealSparseVector data;
  real1 default_value;

  SparseCpuRealStorage(tcapint n)
      : RealStorage(DeviceTag::CPU, n), data(), default_value(ZERO_R1) {}

  bool is_sparse() override { return true; }

  /**
   * Return the sparse element count
   */
  tcapint get_sparse_size() override { return data.size(); }

  /**
   * Get the real element at the position
   */
  real1 operator[](tcapint idx) override {
    const auto it = data.find(idx);
    if (it == data.end()) {
      return default_value;
    }
    return it->second;
  }

  void write(tcapint idx, real1 val) override {
    if (std::abs(val - default_value) <= FP_NORM_EPSILON) {
      data.erase(idx);
    } else {
      data[idx] = val;
    }
  }

  void add(tcapint idx, real1 val) override {
    if (std::abs(val) > FP_NORM_EPSILON) {
      data[idx] += val;
    }
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

    SparseCpuComplexStoragePtr n =
        std::make_shared<SparseCpuComplexStorage>(size);
    n->default_value = default_value;
    for (auto it = data.begin(); it != data.end(); ++it) {
      n->data[it->first] = (complex)it->second;
    }

    return n;
  }

  StoragePtr cpu() override { return get_ptr(); }
  StoragePtr gpu(int64_t did = -1) override {
    throw std::domain_error("Don't use sparse Storage::gpu() (for now)!");
  }
};
typedef std::shared_ptr<SparseCpuRealStorage> SparseCpuRealStoragePtr;
} // namespace Weed
