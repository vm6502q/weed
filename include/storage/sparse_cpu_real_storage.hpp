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

  SparseCpuRealStorage(const RealSparseVector &v, const tcapint &n,
                       const real1 dv = ZERO_R1)
      : RealStorage(DeviceTag::CPU, n), data(v), default_value(dv) {}
  SparseCpuRealStorage(tcapint n)
      : RealStorage(DeviceTag::CPU, n), data(), default_value(ZERO_R1) {}

  bool is_sparse() const override { return default_value == ZERO_R1; }

  /**
   * Return the sparse element count
   */
  tcapint get_sparse_size() const override { return data.size(); }

  /**
   * Get the real element at the position
   */
  real1 operator[](const tcapint &idx) const override {
    const auto it = data.find(idx);
    if (it == data.end()) {
      return default_value;
    }
    return it->second;
  }

  void write(const tcapint &idx, const real1 &val) override {
    if (std::abs(val - default_value) <= FP_NORM_EPSILON) {
      data.erase(idx);
    } else {
      data[idx] = val;
    }
  }

  void add(const tcapint &idx, const real1 &val) override {
    if (std::abs(val - default_value) <= FP_NORM_EPSILON) {
      return;
    }

    auto it = data.find(idx);

    if (it == data.end()) {
      data[idx] = val;
      return;
    }

    if (std::abs(val + it->second + default_value) <= FP_NORM_EPSILON) {
      data.erase(it);
      return;
    }

    it->second += val;
  }

  void FillZeros() override { FillValue(ZERO_R1); }
  void FillOnes() override { FillValue(ONE_R1); }
  void FillValue(const real1 &v) override {
    data.clear();
    default_value = v;
  }

  StoragePtr Upcast(const DType &dt) override {
    if (dt == DType::REAL) {
      return get_ptr();
    }

    SparseCpuComplexStoragePtr n =
        std::make_shared<SparseCpuComplexStorage>(size);
    for (auto it = data.begin(); it != data.end(); ++it) {
      n->data[it->first] = (complex)it->second;
    }

    return n;
  }

  StoragePtr cpu() override { return get_ptr(); }
  StoragePtr gpu(const int64_t &did = -1) override {
    throw std::domain_error("Don't use sparse Storage::gpu() (for now)!");
  }
};
typedef std::shared_ptr<SparseCpuRealStorage> SparseCpuRealStoragePtr;
} // namespace Weed
