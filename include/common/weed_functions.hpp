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
#if WEED_CPP_STD >= 20
#include <bit>
#endif

namespace Weed {
inline tlenint log2Gpu(tcapint n) {
// Source:
// https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers#answer-11376759
#if WEED_CPP_STD >= 20
  return std::bit_width(n) - 1U;
#elif ENABLE_INTRINSICS && defined(_WIN32) && !defined(__CYGWIN__)
#if WEED_TCAPPOW < 6
  return (tlenint)(bitsInByte * sizeof(unsigned int) -
                   _lzcnt_u32((unsigned int)n) - 1U);
#else
  return (tlenint)(bitsInByte * sizeof(unsigned long long) -
                   _lzcnt_u64((unsigned long long)n) - 1U);
#endif
#elif ENABLE_INTRINSICS && !defined(__APPLE__)
#if WEED_TCAPPOW < 6
  return (tlenint)(bitsInByte * sizeof(unsigned int) -
                   __builtin_clz((unsigned int)n) - 1U);
#else
  return (tlenint)(bitsInByte * sizeof(unsigned long long) -
                   __builtin_clzll((unsigned long long)n) - 1U);
#endif
#else
  tlenint pow = 0U;
  tcapint p = n >> 1U;
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
} // namespace Weed
