//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of
// Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#pragma once

#include "weed_types.hpp"

#ifdef __APPLE__
#include "TargetConditionals.h"
#elif ENABLE_INTRINSICS
#include "immintrin.h"
#endif

#include <set>
#include <vector>
#if CPP_STD >= 20
#include <bit>
#endif

#define _bi_compare(left, right)                                               \
  if (left > right) {                                                          \
    return 1;                                                                  \
  }                                                                            \
  if (left < right) {                                                          \
    return -1;                                                                 \
  }                                                                            \
                                                                               \
  return 0;

#if (QBCAPPOW < 7) || ((QBCAPPOW < 8) && defined(__SIZEOF_INT128__)) ||        \
    ((QBCAPPOW > 7) && defined(BOOST_AVAILABLE))
inline void bi_not_ip(vecCapInt *left) { *left = ~(*left); }
inline void bi_and_ip(vecCapInt *left, const vecCapInt &right) {
  *left &= right;
}
inline void bi_or_ip(vecCapInt *left, const vecCapInt &right) {
  *left |= right;
}
inline void bi_xor_ip(vecCapInt *left, const vecCapInt &right) {
  *left ^= right;
}
inline double bi_to_double(const vecCapInt &in) { return (double)in; }

inline void bi_increment(vecCapInt *pBigInt, const vecCapInt &value) {
  *pBigInt += value;
}
inline void bi_decrement(vecCapInt *pBigInt, const vecCapInt &value) {
  *pBigInt -= value;
}

inline void bi_lshift_ip(vecCapInt *left, const size_t &right) {
  *left <<= right;
}
inline void bi_rshift_ip(vecCapInt *left, const size_t &right) {
  *left >>= right;
}

inline int bi_and_1(const vecCapInt &left) { return (bool)(left & 1); }

inline int bi_compare(const vecCapInt &left, const vecCapInt &right) {
  _bi_compare(left, right)
}
inline int bi_compare_0(const vecCapInt &left) { return (int)(bool)left; }
inline int bi_compare_1(const vecCapInt &left) { _bi_compare(left, 1U); }

inline void bi_add_ip(vecCapInt *left, const vecCapInt &right) {
  *left += right;
}
inline void bi_sub_ip(vecCapInt *left, const vecCapInt &right) {
  *left -= right;
}

inline void bi_div_mod(const vecCapInt &left, const vecCapInt &right,
                       vecCapInt *quotient, vecCapInt *rmndr) {
  if (quotient) {
    *quotient = left / right;
  }
  if (rmndr) {
    *rmndr = left % right;
  }
}
#ifdef __SIZEOF_INT128__
inline void bi_div_mod_small(const vecCapInt &left, uint64_t right,
                             vecCapInt *quotient, uint64_t *rmndr) {
  if (quotient) {
    *quotient = left / right;
  }
  if (rmndr) {
    *rmndr = (uint64_t)(left % right);
  }
}
#else
inline void bi_div_mod_small(const vecCapInt &left, uint32_t right,
                             vecCapInt *quotient, uint32_t *rmndr) {
  if (quotient) {
    *quotient = left / right;
  }
  if (rmndr) {
    *rmndr = (uint32_t)(left % right);
  }
}
#endif
#endif

namespace Weed {

inline vecLenInt log2Gpu(vecCapIntGpu n) {
// Source:
// https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers#answer-11376759
#if CPP_STD >= 20
  return std::bit_width(n) - 1U;
#elif ENABLE_INTRINSICS && defined(_WIN32) && !defined(__CYGWIN__)
#if UINTPOW < 6
  return (vecLenInt)(bitsInByte * sizeof(unsigned int) -
                     _lzcnt_u32((unsigned int)n) - 1U);
#else
  return (vecLenInt)(bitsInByte * sizeof(unsigned long long) -
                     _lzcnt_u64((unsigned long long)n) - 1U);
#endif
#elif ENABLE_INTRINSICS && !defined(__APPLE__)
#if UINTPOW < 6
  return (vecLenInt)(bitsInByte * sizeof(unsigned int) -
                     __builtin_clz((unsigned int)n) - 1U);
#else
  return (vecLenInt)(bitsInByte * sizeof(unsigned long long) -
                     __builtin_clzll((unsigned long long)n) - 1U);
#endif
#else
  vecLenInt pow = 0U;
  vecCapIntGpu p = n >> 1U;
  while (p) {
    p >>= 1U;
    ++pow;
  }
  return pow;
#endif
}

inline vecLenInt popCountGpu(vecCapIntGpu n) {
#if CPP_STD >= 20
  return (vecLenInt)std::popcount(n);
#elif (defined(__GNUC__) || defined(__clang__)) && !TARGET_OS_IPHONE &&        \
    !TARGET_IPHONE_SIMULATOR
  return __builtin_popcount(n);
#else
  vecLenInt popCount;
  for (popCount = 0U; n; ++popCount) {
    n &= n - 1U;
  }
  return popCount;
#endif
}

#if (QBCAPPOW < 7) || ((QBCAPPOW < 8) && defined(__SIZEOF_INT128__)) ||        \
    ((QBCAPPOW > 7) && defined(BOOST_AVAILABLE))
inline int bi_log2(const vecCapInt &n) { return log2Gpu((vecCapIntGpu)n); }
#endif
inline vecLenInt log2(vecCapInt n) { return (vecLenInt)bi_log2(n); }

inline vecCapInt pow2(const vecLenInt &p) { return ONE_VCI << p; }
inline vecCapIntGpu pow2Gpu(const vecLenInt &p) {
  return (vecCapIntGpu)1U << p;
}
inline vecCapInt pow2Mask(const vecLenInt &p) {
  vecCapInt toRet = ONE_VCI << p;
  bi_decrement(&toRet, 1U);
  return toRet;
}
inline vecCapIntGpu pow2MaskGpu(const vecLenInt &p) {
  return ((vecCapIntGpu)1U << p) - 1U;
}
inline vecCapInt bitSlice(const vecLenInt &bit, const vecCapInt &source) {
  return (ONE_VCI << bit) & source;
}
inline vecCapIntGpu bitSliceGpu(const vecLenInt &bit,
                                const vecCapIntGpu &source) {
  return ((vecCapIntGpu)1U << bit) & source;
}
inline vecCapInt bitRegMask(const vecLenInt &start, const vecLenInt &length) {
  vecCapInt toRet = ONE_VCI << length;
  bi_decrement(&toRet, 1U);
  bi_lshift_ip(&toRet, start);
  return toRet;
}
inline vecCapIntGpu bitRegMaskGpu(const vecLenInt &start,
                                  const vecLenInt &length) {
  return (((vecCapIntGpu)1U << length) - 1U) << start;
}
// Source:
// https://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
inline bool isPowerOfTwo(const vecCapInt &x) {
  vecCapInt y = x;
  bi_decrement(&y, 1U);
  bi_and_ip(&y, x);
  return (bi_compare_0(x) != 0) && (bi_compare_0(y) == 0);
}
inline bool isPowerOfTwoGpu(const vecCapIntGpu &x) {
  return x && !(x & (x - 1U));
}
inline bool isBadBitRange(const vecLenInt &start, const vecLenInt &length,
                          const vecLenInt &qubitCount) {
  return ((start + length) > qubitCount) ||
         ((vecLenInt)(start + length) < start);
}
inline bool isBadPermRange(const vecCapIntGpu &start,
                           const vecCapIntGpu &length,
                           const vecCapIntGpu &maxQPowerGpu) {
  return ((start + length) > maxQPowerGpu) ||
         ((vecCapIntGpu)(start + length) < start);
}
inline void ThrowIfQbIdArrayIsBad(const std::vector<vecLenInt> &controls,
                                  const vecLenInt &qubitCount,
                                  std::string message) {
  std::set<vecLenInt> dupes;
  for (const vecLenInt &control : controls) {
    if (control >= qubitCount) {
      throw std::invalid_argument(message);
    }

    if (dupes.find(control) == dupes.end()) {
      dupes.insert(control);
    } else {
      throw std::invalid_argument(message +
                                  " (Found duplicate qubit indices!)");
    }
  }
}

// These are utility functions defined in qinterface/protected.cpp:
unsigned char *cl_alloc(size_t ucharCount);
void cl_free(void *toFree);
void mul2x2(const complex *left, const complex *right, complex *out);
void exp2x2(const complex *matrix2x2, complex *outMatrix2x2);
void log2x2(const complex *matrix2x2, complex *outMatrix2x2);
void inv2x2(const complex *matrix2x2, complex *outMatrix2x2);
bool isOverflowAdd(vecCapIntGpu inOutInt, vecCapIntGpu inInt,
                   const vecCapIntGpu &signMask,
                   const vecCapIntGpu &lengthPower);
bool isOverflowSub(vecCapIntGpu inOutInt, vecCapIntGpu inInt,
                   const vecCapIntGpu &signMask,
                   const vecCapIntGpu &lengthPower);
vecCapInt pushApartBits(const vecCapInt &perm,
                        const std::vector<vecCapInt> &skipPowers);
vecCapInt intPow(const vecCapInt &base, const vecCapInt &power);
vecCapIntGpu intPowGpu(vecCapIntGpu base, vecCapIntGpu power);

#if QBCAPPOW > 6
std::ostream &operator<<(std::ostream &os, const vecCapInt &b);
std::istream &operator>>(std::istream &is, vecCapInt &b);
#endif

#if ENABLE_ENV_VARS
const vecLenInt WEED_MAX_CPU_POW_DEFAULT =
    getenv("WEED_MAX_CPU_POW")
        ? (vecLenInt)std::stoi(std::string(getenv("WEED_MAX_CPU_POW")))
        : -1;
const vecLenInt WEED_MAX_PAGE_POW_DEFAULT =
    getenv("WEED_MAX_PAGE_POW")
        ? (vecLenInt)std::stoi(std::string(getenv("WEED_MAX_PAGE_POW")))
        : WEED_MAX_CPU_POW_DEFAULT;
const vecLenInt WEED_MAX_PAGING_POW_DEFAULT =
    getenv("WEED_MAX_PAGING_POW")
        ? (vecLenInt)std::stoi(std::string(getenv("WEED_MAX_PAGING_POW")))
        : WEED_MAX_CPU_POW_DEFAULT;
const vecLenInt PSTRIDEPOW_DEFAULT =
    (vecLenInt)(getenv("WEED_PSTRIDEPOW")
                    ? std::stoi(std::string(getenv("WEED_PSTRIDEPOW")))
                    : PSTRIDEPOW);
const size_t WEED_SPARSE_MAX_ALLOC_MB_DEFAULT =
    (size_t)(getenv("WEED_SPARSE_MAX_ALLOC_MB")
                 ? std::stoi(std::string(getenv("WEED_SPARSE_MAX_ALLOC_MB")))
                 : -1);
const real1_f _weed_sparse_thresh =
    getenv("WEED_SPARSE_TRUNCATION_THRESHOLD")
        ? (real1_f)std::stof(
              std::string(getenv("WEED_SPARSE_TRUNCATION_THRESHOLD")))
        : REAL1_EPSILON;
#else
const vecLenInt WEED_MAX_CPU_POW_DEFAULT = -1;
const vecLenInt WEED_MAX_PAGE_POW_DEFAULT = WEED_MAX_CPU_POW_DEFAULT;
const vecLenInt WEED_MAX_PAGING_POW_DEFAULT = WEED_MAX_CPU_POW_DEFAULT;
const vecLenInt PSTRIDEPOW_DEFAULT = PSTRIDEPOW;
const size_t WEED_SPARSE_MAX_ALLOC_MB_DEFAULT = -1;
const real1_f _weed_sparse_thresh = REAL1_EPSILON;
#endif
const size_t WEED_SPARSE_MAX_ALLOC_BYTES_DEFAULT =
    WEED_SPARSE_MAX_ALLOC_MB_DEFAULT * 1024U * 1024U;
const size_t WEED_SPARSE_MAX_KEYS =
    (WEED_SPARSE_MAX_ALLOC_BYTES_DEFAULT / SPARSE_KEY_BYTES) >> 1U;
} // namespace Weed
