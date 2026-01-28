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

#include "weed_functions.hpp"

#include <functional>

namespace Weed {

// Called once per value between begin and end.
typedef std::function<void(const tcapint &, const unsigned &cpu)> ParallelFunc;
typedef std::function<tcapint(const tcapint &)> IncrementFunc;

class ParallelFor {
private:
  const tcapint pStride;
  tlenint dispatchThreshold;
  unsigned numCores;

public:
  ParallelFor();

  unsigned GetNumCores() { return numCores; }

  void SetConcurrencyLevel(unsigned num) {
    if (!num) {
      num = 1U;
    }
    if (numCores == num) {
      return;
    }
    numCores = num;
    const tlenint pStridePow = log2Gpu(pStride);
    const tlenint minStridePow = (tlenint)pow2Gpu(log2Gpu(numCores - 1U));
    dispatchThreshold =
        (pStridePow > minStridePow) ? (pStridePow - minStridePow) : 0U;
  }
  unsigned GetConcurrencyLevel() { return numCores; }
  tcapint GetStride() { return pStride; }
  tlenint GetPreferredConcurrencyPower() { return dispatchThreshold; }
  /*
   * Parallelization routines for spreading work across multiple cores.
   */

  /**
   * Iterate through the permutations a maximum of end-begin times, allowing
   * the caller to control the incrementation offset through 'inc'.
   */
  void par_for_inc(const tcapint begin, const tcapint itemCount, IncrementFunc,
                   ParallelFunc fn);

  /**
   * Call fn once for every numerical value between begin and end.
   */
  void par_for(const tcapint begin, const tcapint end, ParallelFunc fn);

  /**
   * Call fn once for every value in a sparse map.
   */
  void par_for(const std::map<tcapint, real1> &sparseMap, ParallelFunc fn);
  /**
   * Call fn once for every value in a sparse map.
   */
  void par_for(const std::map<tcapint, complex> &sparseMap, ParallelFunc fn);
  /**
   * Call fn once for every value in a sparse set.
   */
  void par_for(const std::set<tcapint> &sparseMap, ParallelFunc fn);
};

extern ParallelFor pfControl;
} // namespace Weed
