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

#include "storage/cpu_complex_storage.hpp"
#include "storage/cpu_int_storage.hpp"
#include "storage/cpu_real_storage.hpp"
#include "storage/sparse_cpu_complex_storage.hpp"
#include "storage/sparse_cpu_real_storage.hpp"
#if ENABLE_GPU
#include "storage/gpu_complex_storage.hpp"
#include "storage/gpu_int_storage.hpp"
#include "storage/gpu_real_storage.hpp"
#endif

namespace Weed {
void Storage::save(std::ostream &os) const {
  write_storage_type(os, stype);
  write_tcapint(os, size);
  // Needs the inheriting struct to do the rest
}

StoragePtr Storage::load(std::istream &is) {
  StorageType stype;
  read_storage_type(is, stype);

  tcapint size;
  read_tcapint(is, size);

  switch (stype) {
  case StorageType::REAL_CPU_DENSE: {
    std::vector<real1> v(size);
    for (tcapint i = 0U; i < size; ++i) {
      read_real(is, v[i]);
    }
    return std::make_shared<CpuRealStorage>(v);
  }
  case StorageType::REAL_GPU_DENSE: {
    std::vector<real1> v(size);
    for (tcapint i = 0U; i < size; ++i) {
      read_real(is, v[i]);
    }
    return std::make_shared<GpuRealStorage>(v);
  }
  case StorageType::COMPLEX_CPU_DENSE: {
    std::vector<complex> v(size);
    for (tcapint i = 0U; i < size; ++i) {
      read_complex(is, v[i]);
    }
    return std::make_shared<CpuComplexStorage>(v);
  }
  case StorageType::COMPLEX_GPU_DENSE: {
    std::vector<complex> v(size);
    for (tcapint i = 0U; i < size; ++i) {
      read_complex(is, v[i]);
    }
    return std::make_shared<GpuComplexStorage>(v);
  }
  case StorageType::INT_CPU_DENSE: {
    std::vector<symint> v(size);
    for (tcapint i = 0U; i < size; ++i) {
      read_symint(is, v[i]);
    }
    return std::make_shared<CpuIntStorage>(v);
  }
  case StorageType::INT_GPU_DENSE: {
    std::vector<symint> v(size);
    for (tcapint i = 0U; i < size; ++i) {
      read_symint(is, v[i]);
    }
    return std::make_shared<GpuIntStorage>(v);
  }
  case StorageType::REAL_CPU_SPARSE: {
    tcapint ksize;
    read_tcapint(is, ksize);
    RealSparseVector s;
    tcapint k;
    real1 v;
    for (tcapint i = 0U; i < ksize; ++i) {
      read_tcapint(is, k);
      read_real(is, v);
      s[k] = v;
    }
    return std::make_shared<SparseCpuRealStorage>(s, size);
  }
  case StorageType::COMPLEX_CPU_SPARSE: {
    tcapint ksize;
    read_tcapint(is, ksize);
    ComplexSparseVector s;
    tcapint k;
    complex v;
    for (tcapint i = 0U; i < ksize; ++i) {
      read_tcapint(is, k);
      read_complex(is, v);
      s[k] = v;
    }
    return std::make_shared<SparseCpuComplexStorage>(s, size);
  }
  case StorageType::NONE_STORAGE_TYPE:
  default:
    throw std::domain_error("Can't recognize StorageType in Storage::load!");
  }

  return nullptr;
}
} // namespace Weed
