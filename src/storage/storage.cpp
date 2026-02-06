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

#include "enums/storage_type.hpp"

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
StoragePtr Storage::load(std::istream &is) {
  char stype_char;
  is >> stype_char;
  const StorageType stype = (StorageType)stype_char;

  uint64_t size_i64;
  is >> size_i64;
  // const tcapint size = (tcapint)size_i64;

  switch (stype) {
  case StorageType::REAL_CPU_DENSE:
    break;
  case StorageType::REAL_GPU_DENSE:
    break;
  case StorageType::COMPLEX_CPU_DENSE:
    break;
  case StorageType::COMPLEX_GPU_DENSE:
    break;
  case StorageType::INT_CPU_DENSE:
    break;
  case StorageType::INT_GPU_DENSE:
    break;
  case StorageType::REAL_CPU_SPARSE:
    break;
  case StorageType::COMPLEX_CPU_SPARSE:
    break;
  case StorageType::NONE_STORAGE_TYPE:
  default:
    throw std::domain_error("Can't recognize StorageType in Storage::load!");
  }

  return nullptr;
}
} // namespace Weed
