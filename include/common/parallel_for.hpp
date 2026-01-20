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
typedef std::function<void(const vecCapIntGpu &, const unsigned &cpu)>
    ParallelFunc;
typedef std::function<vecCapIntGpu(const vecCapIntGpu &)> IncrementFunc;

class ParallelFor {
private:
  const vecCapIntGpu pStride;
  vecLenInt dispatchThreshold;
  unsigned numCores;

public:
  ParallelFor();

  void SetConcurrencyLevel(unsigned num) {
    if (!num) {
      num = 1U;
    }
    if (numCores == num) {
      return;
    }
    numCores = num;
    const vecLenInt pStridePow = log2Gpu(pStride);
    const vecLenInt minStridePow = (vecLenInt)pow2Gpu(log2Gpu(numCores - 1U));
    dispatchThreshold =
        (pStridePow > minStridePow) ? (pStridePow - minStridePow) : 0U;
  }
  unsigned GetConcurrencyLevel() { return numCores; }
  vecCapIntGpu GetStride() { return pStride; }
  vecLenInt GetPreferredConcurrencyPower() { return dispatchThreshold; }
  /*
   * Parallelization routines for spreading work across multiple cores.
   */

  /**
   * Iterate through the permutations a maximum of end-begin times, allowing
   * the caller to control the incrementation offset through 'inc'.
   */
  void par_for_inc(const vecCapIntGpu begin, const vecCapIntGpu itemCount,
                   IncrementFunc, ParallelFunc fn);

  /** Call fn once for every numerical value between begin and end. */
  void par_for(const vecCapIntGpu begin, const vecCapIntGpu end,
               ParallelFunc fn);

  /**
   * Skip over the skipPower bits.
   *
   * For example, if skipPower is 2, it will count:
   *   0000, 0001, 0100, 0101, 1000, 1001, 1100, 1101.
   *     ^     ^     ^     ^     ^     ^     ^     ^ - The second bit is
   *                                                   untouched.
   */
  void par_for_skip(const vecCapIntGpu begin, const vecCapIntGpu end,
                    const vecCapIntGpu skipPower, const vecLenInt skipBitCount,
                    ParallelFunc fn);

  /** Skip over the bits listed in maskArray in the same fashion as
   * par_for_skip. */
  void par_for_mask(const vecCapIntGpu, const vecCapIntGpu,
                    const std::vector<vecCapIntGpu> &maskArray,
                    ParallelFunc fn);

  /** Iterate over a sparse state vector. */
  void par_for_set(const std::set<vecCapIntGpu> &sparseSet, ParallelFunc fn);

  /** Iterate over a sparse state vector. */
  void par_for_set(const std::vector<vecCapIntGpu> &sparseSet, ParallelFunc fn);

  /** Iterate over the power set of 2 sparse state vectors. */
  void par_for_sparse_compose(const std::vector<vecCapIntGpu> &lowSet,
                              const std::vector<vecCapIntGpu> &highSet,
                              const vecLenInt &highStart, ParallelFunc fn);
};

} // namespace Weed
