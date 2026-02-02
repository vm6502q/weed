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
 * Storage for real data type elements
 */
struct RealStorage : Storage {
  RealStorage(const DeviceTag &dtg, const tcapint &n)
      : Storage(dtg, DType::REAL, n) {}

  virtual ~RealStorage() {}

  /**
   * Get the real element at the position
   */
  virtual real1 operator[](const tcapint &idx) const = 0;

  /**
   * Set the real element at the position
   */
  virtual void write(const tcapint &idx, const real1 &val) = 0;

  /**
   * Add to the real element at the position
   */
  virtual void add(const tcapint &idx, const real1 &val) = 0;

  /**
   * Fill the entire Storage with specified real value
   */
  virtual void FillValue(const real1 &v) = 0;

#if defined(__APPLE__)
  static real1 *_aligned_state_vec_alloc(tcapint allocSize) {
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

  static std::unique_ptr<real1[], void (*)(real1 *)> Alloc(tcapint elemCount) {
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
typedef std::shared_ptr<RealStorage> RealStoragePtr;
} // namespace Weed
