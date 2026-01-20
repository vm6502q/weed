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

#include "real_storage.hpp"

namespace Weed {
struct CpuRealStorage : RealStorage {
  RealPtr data;

  CpuRealStorage(vecCapIntGpu n)
      : RealStorage(DeviceTag::CPU, n), data(Alloc(n)) {
    std::fill(data.get(), data.get() + n, ZERO_R1);
  }

  CpuRealStorage(std::vector<real1> i)
      : RealStorage(DeviceTag::CPU, i.size()), data(Alloc(i.size())) {
    std::copy(i.begin(), i.end(), data.get());
  }

  ~CpuRealStorage() {}
};
} // namespace Weed
