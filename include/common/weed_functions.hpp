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

namespace Weed {
inline tlenint log2Gpu(tcapint n) {
// Source:
// https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers#answer-11376759
#if CPP_STD >= 20
  return std::bit_width(n) - 1U;
#elif ENABLE_INTRINSICS && defined(_WIN32) && !defined(__CYGWIN__)
#if UINTPOW < 6
  return (tlenint)(bitsInByte * sizeof(unsigned int) -
                   _lzcnt_u32((unsigned int)n) - 1U);
#else
  return (tlenint)(bitsInByte * sizeof(unsigned long long) -
                   _lzcnt_u64((unsigned long long)n) - 1U);
#endif
#elif ENABLE_INTRINSICS && !defined(__APPLE__)
#if UINTPOW < 6
  return (tlenint)(bitsInByte * sizeof(unsigned int) -
                   __builtin_clz((unsigned int)n) - 1U);
#else
  return (tlenint)(bitsInByte * sizeof(unsigned long long) -
                   __builtin_clzll((unsigned long long)n) - 1U);
#endif
#else
  tlenint pow = 0U;
  bitCapIntOcl p = n >> 1U;
  while (p) {
    p >>= 1U;
    ++pow;
  }
  return pow;
#endif
}
inline tcapint pow2Gpu(const tlenint &p) { return (tcapint)1U << p; }

// These are utility functions defined in qinterface/protected.cpp:
unsigned char *cl_alloc(size_t ucharCount);
void cl_free(void *toFree);

#if ENABLE_ENV_VARS
const tlenint WEED_MAX_CPU_POW_DEFAULT =
    getenv("WEED_MAX_CPU_POW")
        ? (tlenint)std::stoi(std::string(getenv("WEED_MAX_CPU_POW")))
        : -1;
const tlenint WEED_MAX_PAGE_POW_DEFAULT =
    getenv("WEED_MAX_PAGE_POW")
        ? (tlenint)std::stoi(std::string(getenv("WEED_MAX_PAGE_POW")))
        : WEED_MAX_CPU_POW_DEFAULT;
const tlenint WEED_MAX_PAGING_POW_DEFAULT =
    getenv("WEED_MAX_PAGING_POW")
        ? (tlenint)std::stoi(std::string(getenv("WEED_MAX_PAGING_POW")))
        : WEED_MAX_CPU_POW_DEFAULT;
const tlenint PSTRIDEPOW_DEFAULT =
    (tlenint)(getenv("WEED_PSTRIDEPOW")
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
const tlenint WEED_MAX_CPU_POW_DEFAULT = -1;
const tlenint WEED_MAX_PAGE_POW_DEFAULT = WEED_MAX_CPU_POW_DEFAULT;
const tlenint WEED_MAX_PAGING_POW_DEFAULT = WEED_MAX_CPU_POW_DEFAULT;
const tlenint PSTRIDEPOW_DEFAULT = PSTRIDEPOW;
const size_t WEED_SPARSE_MAX_ALLOC_MB_DEFAULT = -1;
const real1_f _weed_sparse_thresh = REAL1_EPSILON;
#endif
const size_t WEED_SPARSE_MAX_ALLOC_BYTES_DEFAULT =
    WEED_SPARSE_MAX_ALLOC_MB_DEFAULT * 1024U * 1024U;
const size_t WEED_SPARSE_MAX_KEYS =
    (WEED_SPARSE_MAX_ALLOC_BYTES_DEFAULT / SPARSE_KEY_BYTES) >> 1U;
} // namespace Weed
