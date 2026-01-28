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

  SparseCpuComplexStorage(tcapint n)
      : ComplexStorage(DeviceTag::CPU, n, true), default_value(ZERO_R1) {}

  /**
   * Get the complex element at the position
   */
  complex operator[](tcapint idx) override {
    const auto it = data.find(idx);
    if (it == data.end()) {
      return default_value;
    }
    return it->second;
  }

  void write(tcapint idx, complex val) override {
    if (std::abs(val - default_value) <= FP_NORM_EPSILON) {
      data.erase(idx);
    } else {
      data[idx] = val;
    }
  }

  void add(tcapint idx, complex val) override {
    if (std::abs(val) > FP_NORM_EPSILON) {
      data[idx] += val;
    }
  }

  void FillZeros() override {
    data.clear();
    default_value = ZERO_CMPLX;
  }
  void FillOnes() override {
    data.clear();
    default_value = ONE_CMPLX;
  }
  void FillValue(complex v) override {
    data.clear();
    default_value = v;
  }

  StoragePtr Upcast(DType dt) override { return get_ptr(); }

  StoragePtr cpu() override { return get_ptr(); }
  StoragePtr gpu(int64_t did = -1) override {
    throw std::domain_error("Don't use sparse Storage::gpu() (for now)!");
  }
};
typedef std::shared_ptr<SparseCpuComplexStorage> SparseCpuComplexStoragePtr;
} // namespace Weed
