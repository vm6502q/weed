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

#include "storage/storage.hpp"

namespace Weed {
/**
 * CPU-accessible storage for real data type elements
 */
struct SparseCpuComplexStorage : ComplexStorage {
  ComplexSparseVector data;
  complex default_value;

  SparseCpuComplexStorage(const ComplexSparseVector &v, const tcapint &n)
      : ComplexStorage(DeviceTag::CPU, n), data(v) {}
  SparseCpuComplexStorage(const tcapint &n)
      : ComplexStorage(DeviceTag::CPU, n), data() {}

  bool is_sparse() const override { return default_value == ZERO_CMPLX; }

  /**
   * Return the sparse element count
   */
  tcapint get_sparse_size() const override { return data.size(); }

  /**
   * Get the complex element at the position
   */
  complex operator[](const tcapint &idx) const override {
    const auto it = data.find(idx);
    if (it == data.end()) {
      return default_value;
    }
    return it->second;
  }

  void write(const tcapint &idx, const complex &val) override {
    if (std::abs(val - default_value) <= FP_NORM_EPSILON) {
      data.erase(idx);
    } else {
      data[idx] = val;
    }
  }

  void add(const tcapint &idx, const complex &val) override {
    if (std::abs(val - default_value) <= FP_NORM_EPSILON) {
      return;
    }

    auto it = data.find(idx);

    if (it == data.end()) {
      data[idx] = val;
      return;
    }

    if (std::abs(val + it->second - default_value) <= FP_NORM_EPSILON) {
      data.erase(it);
      return;
    }

    it->second += val;
  }

  void FillZeros() override { FillValue(ZERO_CMPLX); }
  void FillOnes() override { FillValue(ONE_CMPLX); }
  void FillValue(const complex &v) override {
    data.clear();
    default_value = v;
  }

  StoragePtr Upcast(const DType &dt) override { return get_ptr(); }

  StoragePtr cpu() override { return get_ptr(); }
  StoragePtr gpu(const int64_t &did = -1) override {
    throw std::domain_error("Don't use sparse Storage::gpu() (for now)!");
  }
};
typedef std::shared_ptr<SparseCpuComplexStorage> SparseCpuComplexStoragePtr;
} // namespace Weed
