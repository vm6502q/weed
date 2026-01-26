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

#include "weed_functions.hpp"

#if ENABLE_COMPLEX_X2
#if FPPOW == 5
#include "common/complex8x2simd.hpp"
#elif FPPOW == 6
#include "common/complex16x2simd.hpp"
#endif
#endif

#include <algorithm>

namespace Weed {

unsigned char *cl_alloc(size_t ucharCount) {
#if defined(__APPLE__)
  void *toRet;
  posix_memalign(&toRet, WEED_ALIGN_SIZE,
                 ((sizeof(unsigned char) * ucharCount) < WEED_ALIGN_SIZE)
                     ? WEED_ALIGN_SIZE
                     : (sizeof(unsigned char) * ucharCount));
  return (unsigned char *)toRet;
#elif defined(_WIN32) && !defined(__CYGWIN__)
  return (unsigned char *)_aligned_malloc(
      ((sizeof(unsigned char) * ucharCount) < WEED_ALIGN_SIZE)
          ? WEED_ALIGN_SIZE
          : (sizeof(unsigned char) * ucharCount),
      WEED_ALIGN_SIZE);
#elif defined(__ANDROID__)
  return (unsigned char *)malloc(sizeof(unsigned char) * ucharCount);
#else
  return (unsigned char *)aligned_alloc(
      WEED_ALIGN_SIZE, ((sizeof(unsigned char) * ucharCount) < WEED_ALIGN_SIZE)
                           ? WEED_ALIGN_SIZE
                           : (sizeof(unsigned char) * ucharCount));
#endif
}

void cl_free(void *toFree) {
#if defined(_WIN32) && !defined(__CYGWIN__)
  _aligned_free(toFree);
#else
  free(toFree);
#endif
}

#if QBCAPPOW > 6
std::ostream &operator<<(std::ostream &os, const vecCapInt &b) {
  if (bi_compare_0(b) == 0) {
    os << "0";

    return os;
  }

  // Calculate the base-10 digits, from lowest to highest.
  std::vector<std::string> digits;
  vecCapInt _b = b;
  while (bi_compare_0(b) != 0) {
    vecCapInt quo;
#ifdef __SIZEOF_INT128__
    uint64_t rem;
#else
    uint32_t rem;
#endif
    bi_div_mod_small(_b, 10U, &quo, &rem);
    digits.push_back(std::to_string((unsigned char)rem));
    _b = quo;
  }

  // Reversing order, print the digits from highest to lowest.
  for (size_t i = digits.size() - 1U; i > 0; --i) {
    os << digits[i];
  }
  // Avoid the need for a signed comparison.
  os << digits[0];

  return os;
}

std::istream &operator>>(std::istream &is, vecCapInt &b) {
  // Get the whole input string at once.
  std::string input;
  is >> input;

  // Start the output address value at 0.
  b = ZERO_VCI;
  for (size_t i = 0; i < input.size(); ++i) {
    // Left shift by 1 base-10 digit.
    b = b * 10U;
    // Add the next lowest base-10 digit.
    bi_increment(&b, (input[i] - 48U));
  }

  return is;
}
#endif
} // namespace Weed
