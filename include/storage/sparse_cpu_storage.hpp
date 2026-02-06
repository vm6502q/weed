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

#include "storage/typed_storage.hpp"

namespace Weed {
/**
 * CPU-accessible storage for real data type elements
 */
template <typename T> struct SparseCpuStorage : TypedStorage<T> {
  std::unordered_map<tcapint, T> data;
  T default_value;

  SparseCpuStorage(const StorageType &stp,
                   const std::unordered_map<tcapint, T> &v, const tcapint &n)
      : TypedStorage<T>(stp, DeviceTag::CPU, n), data(v), default_value() {}
  SparseCpuStorage(const StorageType &stp, const tcapint &n)
      : TypedStorage<T>(stp, DeviceTag::CPU, n), data(), default_value() {}

  bool is_sparse() const override { return default_value == T(); }

  /**
   * Return the sparse element count
   */
  tcapint get_sparse_size() const override { return data.size(); }

  /**
   * Get the complex element at the position
   */
  T operator[](const tcapint &idx) const override {
    const auto it = data.find(idx);
    if (it == data.end()) {
      return default_value;
    }
    return it->second;
  }

  void write(const tcapint &idx, const T &val) override {
    if (std::abs(val - default_value) <= REAL1_EPSILON) {
      data.erase(idx);
    } else {
      data[idx] = val;
    }
  }

  void add(const tcapint &idx, const T &val) override {
    if (std::abs(val - default_value) <= REAL1_EPSILON) {
      return;
    }

    auto it = data.find(idx);

    if (it == data.end()) {
      data[idx] = val;
      return;
    }

    if (std::abs(val + it->second - default_value) <= REAL1_EPSILON) {
      data.erase(it);
      return;
    }

    it->second += val;
  }

  void FillValue(const T &v) override {
    data.clear();
    default_value = v;
  }

  StoragePtr Upcast(const DType &dt) override {
    return TypedStorage<T>::get_ptr();
  }

  StoragePtr cpu() override { return TypedStorage<T>::get_ptr(); }
  StoragePtr gpu(const int64_t &did = -1) override = 0;
};
} // namespace Weed
