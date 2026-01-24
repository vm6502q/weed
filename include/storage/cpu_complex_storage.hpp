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

namespace Weed {
/**
 * CPU-accessible storage for complex data type elements
 */
struct CpuComplexStorage : ComplexStorage {
  ComplexPtr data;

  CpuComplexStorage(vecCapIntGpu n)
      : ComplexStorage(DeviceTag::CPU, n), data(Alloc(n)) {}

  CpuComplexStorage(std::vector<complex> i)
      : ComplexStorage(DeviceTag::CPU, i.size()), data(Alloc(i.size())) {
    std::copy(i.begin(), i.end(), data.get());
  }

  ~CpuComplexStorage() {}

  complex operator[](vecCapInt idx) { return data.get()[(size_t)idx]; }

  void FillZeros() { std::fill(data.get(), data.get() + size, ZERO_CMPLX); }
  void FillOnes() { std::fill(data.get(), data.get() + size, ONE_CMPLX); }

  StoragePtr Upcast(DType dt) { return get_ptr(); };
};
typedef std::shared_ptr<CpuComplexStorage> CpuComplexStoragePtr;
} // namespace Weed
