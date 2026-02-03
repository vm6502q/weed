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

#include <vector>

namespace Weed {
/**
 * CPU-accessible storage
 */
template <typename T> struct CpuStorage : TypedStorage<T> {
  std::unique_ptr<T[], void (*)(T *)> data;
  CpuStorage(const tcapint &n)
      : TypedStorage<T>(DeviceTag::CPU, n), data(TypedStorage<T>::Alloc(n)) {}
  CpuStorage(const std::vector<T> &i)
      : TypedStorage<T>(DeviceTag::CPU, i.size()),
        data(TypedStorage<T>::Alloc(i.size())) {
    std::copy(i.begin(), i.end(), data.get());
  }

  T operator[](const tcapint &idx) const override {
    if (idx >= TypedStorage<T>::size) {
      throw std::invalid_argument(
          "CpuStorage::operator[] argument out-of-bounds!");
    }

    return data.get()[(size_t)idx];
  }

  void write(const tcapint &idx, const T &val) override {
    if (idx >= TypedStorage<T>::size) {
      throw std::invalid_argument(
          "CpuStorage::write(i, v) index out-of-bounds!");
    }

    data.get()[(size_t)idx] = val;
  }

  void add(const tcapint &idx, const T &val) override {
    if (idx >= TypedStorage<T>::size) {
      throw std::invalid_argument("CpuStorage::add(i, v) index out-of-bounds!");
    }

    data.get()[(size_t)idx] += val;
  }

  void FillValue(const T &v) override {
    std::fill(data.get(), data.get() + TypedStorage<T>::size, v);
  }

  virtual StoragePtr Upcast(const DType &dt) = 0;

  StoragePtr cpu() override { return TypedStorage<T>::get_ptr(); }
};
} // namespace Weed
