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

#include "storage.hpp"

namespace Weed {
struct RealStorage : Storage {
  RealStorage(DeviceTag dtg, vecCapInt n) : Storage(dtg, DType::REAL, n) {}

  virtual ~RealStorage() {}

  virtual real1 operator[](vecCapInt idx) = 0;

#if defined(__APPLE__)
  static real1 *_aligned_state_vec_alloc(vecCapIntGpu allocSize) {
    void *toRet;
    posix_memalign(&toRet, WEED_ALIGN_SIZE, allocSize);
    return (real1 *)toRet;
  }
#endif

  static void deleter(real1 *c) {
#if defined(__ANDROID__)
    delete c;
#elif defined(_WIN32) && !defined(__CYGWIN__)
    _aligned_free(c);
#else
    free(c);
#endif
  }

  static std::unique_ptr<real1[], void (*)(real1 *)>
  Alloc(vecCapIntGpu elemCount) {
#if defined(__ANDROID__)
    return std::unique_ptr<real1[], void (*)(real1 *)>(new real1[elemCount],
                                                       deleter);
#else
    size_t allocSize = sizeof(real1) * elemCount;
    if (allocSize < WEED_ALIGN_SIZE) {
      allocSize = WEED_ALIGN_SIZE;
    }
#if defined(__APPLE__)
    return std::unique_ptr<real1[], void (*)(real1 *)>(
        _aligned_state_vec_alloc(allocSize), deleter);
#elif defined(_WIN32) && !defined(__CYGWIN__)
    return std::unique_ptr<real1[], void (*)(real1 *)>(
        (real1 *)_aligned_malloc(allocSize, WEED_ALIGN_SIZE), deleter);
#else
    return std::unique_ptr<real1[], void (*)(real1 *)>(
        (real1 *)aligned_alloc(WEED_ALIGN_SIZE, allocSize), deleter);
#endif
#endif
  }
};
} // namespace Weed
