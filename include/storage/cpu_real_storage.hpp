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

  CpuRealStorage(vecCapIntGpu n)
      : RealStorage(DeviceTag::CPU, n), data(Alloc(n)) {}

  CpuRealStorage(std::vector<real1> i)
      : RealStorage(DeviceTag::CPU, i.size()), data(Alloc(i.size())) {
    std::copy(i.begin(), i.end(), data.get());
  }

  ~CpuRealStorage() {}

  real1 operator[](vecCapInt idx) { return data.get()[(size_t)idx]; }

  void FillZeros() { std::fill(data.get(), data.get() + size, ZERO_R1); }
  void FillOnes() { std::fill(data.get(), data.get() + size, ONE_R1); }

  StoragePtr Upcast(DType dt) {
    if (dt == DType::REAL) {
      return get_ptr();
    }

    CpuComplexStoragePtr n = std::make_shared<CpuComplexStorage>(size);
    std::transform(data.get(), data.get() + size, n->data.get(),
                   [](real1 v) { return complex(v, ZERO_R1); });

    return n;
  }
};
} // namespace Weed
