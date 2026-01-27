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
} // namespace Weed
