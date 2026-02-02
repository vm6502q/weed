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

#include "common/parallel_for.hpp"

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <direct.h>
#endif

#if ENABLE_PTHREAD
#include <atomic>
#include <future>

#define DECLARE_ATOMIC_TCAPINT() std::atomic<tcapint> idx;
#define ATOMIC_ASYNC(...)                                                      \
    std::async(std::launch::async, [__VA_ARGS__]()
#define ATOMIC_INC() i = idx++;
#endif

namespace Weed {
#if ENABLE_ENV_VARS
const tlenint PSTRIDEPOW_DEFAULT =
    (tlenint)(getenv("WEED_PSTRIDEPOW")
                  ? std::stoi(std::string(getenv("WEED_PSTRIDEPOW")))
                  : PSTRIDEPOW);
#else
const tlenint PSTRIDEPOW_DEFAULT = PSTRIDEPOW;
#endif

ParallelFor::ParallelFor()
    : pStride(pow2Gpu((tlenint)PSTRIDEPOW_DEFAULT))
#if ENABLE_PTHREAD
      ,
      numCores(std::thread::hardware_concurrency())
#else
      ,
      numCores(1U)
#endif
{
  const tlenint pStridePow = log2Gpu(pStride);
  const tlenint minStridePow =
      (numCores > 1U) ? (tlenint)pow2Gpu(log2Gpu(numCores - 1U)) : 0U;
  dispatchThreshold =
      (pStridePow > minStridePow) ? (pStridePow - minStridePow) : 0U;
}

void ParallelFor::par_for(const tcapint &begin, const tcapint &end,
                          ParallelFunc fn) {
  par_for_inc(
      begin, end - begin, [](const tcapint &i) { return i; }, fn);
}

void ParallelFor::par_for(const ComplexSparseVector &sparseMap,
                          ParallelFunc fn) {
  par_for_inc(
      0U, sparseMap.size(),
      [&sparseMap](const tcapint &i) {
        auto it = sparseMap.begin();
        std::advance(it, i);
        return it->first;
      },
      fn);
}

void ParallelFor::par_for(const RealSparseVector &sparseMap, ParallelFunc fn) {
  par_for_inc(
      0U, sparseMap.size(),
      [&sparseMap](const tcapint &i) {
        auto it = sparseMap.begin();
        std::advance(it, i);
        return it->first;
      },
      fn);
}

void ParallelFor::par_for(const std::set<tcapint> &sparseSet, ParallelFunc fn) {
  par_for_inc(
      0U, sparseSet.size(),
      [&sparseSet](const tcapint &i) {
        auto it = sparseSet.begin();
        std::advance(it, i);
        return (*it);
      },
      fn);
}

#if ENABLE_PTHREAD
/*
 * Iterate through the permutations a maximum of end-begin times, allowing the
 * caller to control the incrementation offset through 'inc'.
 */
void ParallelFor::par_for_inc(const tcapint &begin, const tcapint &itemCount,
                              IncrementFunc inc, ParallelFunc fn) {
  const tcapint Stride = pStride;
  unsigned threads = (unsigned)(itemCount / pStride);
  if (threads > numCores) {
    threads = numCores;
  }

  if (threads <= 1U) {
    const tcapint maxLcv = begin + itemCount;
    for (tcapint j = begin; j < maxLcv; ++j) {
      fn(inc(j), 0U);
    }

    return;
  }

  DECLARE_ATOMIC_TCAPINT();
  idx = 0U;
  std::vector<std::future<void>> futures;
  futures.reserve(threads);
  for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures.emplace_back(ATOMIC_ASYNC(cpu, &idx, &begin, &itemCount, &Stride, inc, fn) {
      for (;;) {
        tcapint i;
        ATOMIC_INC();
        const tcapint l = i * Stride;
        if (l >= itemCount) {
          break;
        }
        const tcapint maxJ =
            ((l + Stride) < itemCount) ? Stride : (itemCount - l);
        for (tcapint j = 0U; j < maxJ; ++j) {
          fn(inc(begin + j + l), cpu);
        }
      }
        }));
  }

  for (std::future<void> &future : futures) {
    future.get();
  }
}
#else
/*
 * Iterate through the permutations a maximum of end-begin times, allowing the
 * caller to control the incrementation offset through 'inc'.
 */
void ParallelFor::par_for_inc(const tcapint &begin, const tcapint &itemCount,
                              IncrementFunc inc, ParallelFunc fn) {
  const tcapint maxLcv = begin + itemCount;
  for (tcapint j = begin; j < maxLcv; ++j) {
    fn(inc(j), 0U);
  }
}
#endif

ParallelFor pfControl;
} // namespace Weed
