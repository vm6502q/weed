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

#include "cpu_complex_storage.hpp"
#include "real_storage.hpp"

#include <algorithm>

namespace Weed {
struct CpuRealStorage : RealStorage {
  RealPtr data;

  CpuRealStorage(vecCapIntGpu n)
      : RealStorage(DeviceTag::CPU, n), data(Alloc(n)) {}

  CpuRealStorage(std::vector<real1> i)
      : RealStorage(DeviceTag::CPU, i.size()), data(Alloc(i.size())) {
    std::copy(i.begin(), i.end(), data.get());
  }

  ~CpuRealStorage() {}

  void FillZero() { std::fill(data.get(), data.get() + size, ZERO_R1); }

  StoragePtr Upcast(DType dt) {
    if (dt == DType::REAL) {
      return get_ptr();
    }

    CpuComplexStorage n(size);
    std::transform(data.get(), data.get() + size, n.data.get(),
                   [](real1 v) { return complex(v, ZERO_R1); });

    return n.get_ptr();
  };
};
} // namespace Weed
