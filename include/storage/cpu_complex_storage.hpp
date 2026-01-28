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

#include "storage/complex_storage.hpp"

#include <vector>

namespace Weed {
/**
 * CPU-accessible storage for complex data type elements
 */
struct CpuComplexStorage : ComplexStorage {
  ComplexPtr data;

  CpuComplexStorage(tcapint n)
      : ComplexStorage(DeviceTag::CPU, n), data(Alloc(n)) {}

  CpuComplexStorage(std::vector<complex> i)
      : ComplexStorage(DeviceTag::CPU, i.size()), data(Alloc(i.size())) {
    std::copy(i.begin(), i.end(), data.get());
  }

  complex operator[](tcapint idx) override { return data.get()[(size_t)idx]; }

  virtual void write(tcapint idx, complex val) {
    data.get()[(size_t)idx] = val;
  }

  virtual void add(tcapint idx, complex val) { data.get()[(size_t)idx] += val; }

  void FillZeros() override {
    std::fill(data.get(), data.get() + size, ZERO_CMPLX);
  }
  void FillOnes() override {
    std::fill(data.get(), data.get() + size, ONE_CMPLX);
  }
  void FillValue(complex v) override {
    std::fill(data.get(), data.get() + size, v);
  }

  StoragePtr Upcast(DType dt) override { return get_ptr(); };

  StoragePtr cpu() override { return get_ptr(); }
  StoragePtr gpu(int64_t did = -1) override;
};
typedef std::shared_ptr<CpuComplexStorage> CpuComplexStoragePtr;
} // namespace Weed
