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
 * Type-specific Storage template
 */
template <typename T> struct TypedStorage : Storage {
  TypedStorage(const DeviceTag &dtg, const tcapint &n)
      : Storage(dtg, get_dtype(), n) {}

  static DType get_dtype() {
    if (std::is_same<complex, T>::value) {
      return DType::COMPLEX;
    }
    if (std::is_same<real1, T>::value) {
      return DType::REAL;
    }
    if (std::is_same<tcapint, T>::value) {
      return DType::INT;
    }
    return DType::DEFAULT_DTYPE;
  }

  virtual ~TypedStorage() {}

  /**
   * Get the real element at the position
   */
  virtual T operator[](const tcapint &idx) const = 0;

  /**
   * Set the real element at the position
   */
  virtual void write(const tcapint &idx, const T &val) = 0;

  /**
   * Add to the real element at the position
   */
  virtual void add(const tcapint &idx, const T &val) = 0;

  /**
   * Fill the entire Storage with 0
   */
  void FillZeros() { FillValue((T)ZERO_R1); }

  /**
   * Fill the entire Storage with 1
   */
  void FillOnes() { FillValue((T)ONE_R1); }

  /**
   * Fill the entire Storage with specified real value
   */
  virtual void FillValue(const T &v) = 0;

  /**
   * Up-cast data type
   */
  virtual StoragePtr Upcast(const DType &dt) = 0;

  static size_t round_up(size_t n, size_t a) { return ((n + a - 1) / a) * a; }

#if defined(__APPLE__)
  static T *_aligned_state_vec_alloc(tcapint allocSize) {
    void *toRet;
    posix_memalign(&toRet, WEED_ALIGN_SIZE, allocSize);
    return (T *)toRet;
  }
#endif

  static void deleter(T *c) {
#if defined(__ANDROID__)
    delete c;
#elif defined(_WIN32) && !defined(__CYGWIN__)
    _aligned_free(c);
#else
    free(c);
#endif
  }

  static std::unique_ptr<T[], void (*)(T *)> Alloc(tcapint elemCount) {
#if defined(__ANDROID__)
    return std::unique_ptr<T[], void (*)(T *)>(new T[elemCount], deleter);
#else
    size_t allocSize = sizeof(T) * elemCount;
    if (allocSize < WEED_ALIGN_SIZE) {
      allocSize = WEED_ALIGN_SIZE;
    }
#if defined(__APPLE__)
    return std::unique_ptr<T[], void (*)(T *)>(
        _aligned_state_vec_alloc(allocSize), deleter);
#elif defined(_WIN32) && !defined(__CYGWIN__)
    return std::unique_ptr<T[], void (*)(T *)>(
        (T *)_aligned_malloc(allocSize, WEED_ALIGN_SIZE), deleter);
#else
    return std::unique_ptr<T[], void (*)(T *)>(
        (T *)aligned_alloc(WEED_ALIGN_SIZE, allocSize), deleter);
#endif
#endif
  }
};
typedef TypedStorage<symint> IntStorage;
typedef TypedStorage<real1> RealStorage;
typedef TypedStorage<complex> ComplexStorage;
typedef std::shared_ptr<IntStorage> IntStoragePtr;
typedef std::shared_ptr<RealStorage> RealStoragePtr;
typedef std::shared_ptr<ComplexStorage> ComplexStoragePtr;
} // namespace Weed
