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

#include "storage/cpu_complex_storage.hpp"
#include "storage/real_storage.hpp"

#include <algorithm>

namespace Weed {
/**
 * CPU-accessible storage for real data type elements
 */
struct CpuRealStorage : RealStorage {
  RealPtr data;

  CpuRealStorage(tcapint n) : RealStorage(DeviceTag::CPU, n), data(Alloc(n)) {}

  CpuRealStorage(const std::vector<real1> &i)
      : RealStorage(DeviceTag::CPU, i.size()), data(Alloc(i.size())) {
    std::copy(i.begin(), i.end(), data.get());
  }

  real1 operator[](tcapint idx) override { return data.get()[(size_t)idx]; }

  void write(tcapint idx, real1 val) override { data.get()[(size_t)idx] = val; }

  void add(tcapint idx, real1 val) override { data.get()[(size_t)idx] += val; }

  void FillZeros() override {
    std::fill(data.get(), data.get() + size, ZERO_R1);
  }
  void FillOnes() override { std::fill(data.get(), data.get() + size, ONE_R1); }
  void FillValue(real1 v) override {
    std::fill(data.get(), data.get() + size, v);
  }

  StoragePtr Upcast(DType dt) override {
    if (dt == DType::REAL) {
      return get_ptr();
    }

    CpuComplexStoragePtr n = std::make_shared<CpuComplexStorage>(size);
    std::transform(data.get(), data.get() + size, n->data.get(),
                   [](real1 v) { return complex(v, ZERO_R1); });

    return n;
  }

  StoragePtr cpu() override { return get_ptr(); }
  StoragePtr gpu(int64_t did = -1) override;
};
typedef std::shared_ptr<CpuRealStorage> CpuRealStoragePtr;
} // namespace Weed
