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

#include "common/oclengine.hpp"
#include "common/weed_types.hpp"

#include <vector>

namespace Weed {
struct QueueItem {
  OCLAPI api_call;
  size_t workItemCount;
  size_t workItemCount2;
  size_t localGroupSize;
  size_t localGroupSize2;
  size_t deallocSize;
  std::vector<BufferPtr> buffers;
  size_t localBuffSize;

  QueueItem()
      : api_call(), workItemCount(0U), workItemCount2(0U), localGroupSize(0U),
        localGroupSize2(0U), deallocSize(0U), buffers(), localBuffSize(0U) {}

  QueueItem(OCLAPI ac, size_t wic, size_t lgs, size_t ds,
            std::vector<BufferPtr> b, size_t wic2 = 0U, size_t lgs2 = 0U,
            size_t lbs = 0U)
      : api_call(ac), workItemCount(wic), workItemCount2(wic2),
        localGroupSize(lgs), localGroupSize2(lgs2), deallocSize(ds), buffers(b),
        localBuffSize(lbs) {}
};
} // namespace Weed
