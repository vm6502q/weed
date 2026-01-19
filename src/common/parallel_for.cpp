//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "common/parallel_for.hpp"
#include "statevector.hpp"

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <direct.h>
#endif

#if ENABLE_PTHREAD
#include <atomic>
#include <future>

#define DECLARE_ATOMIC_vecCapInt() std::atomic<vecCapIntGpu> idx;
#define ATOMIC_ASYNC(...)                                                                                              \
    std::async(std::launch::async, [__VA_ARGS__]()
#define ATOMIC_INC() i = idx++;
#endif

namespace Weed {

ParallelFor::ParallelFor()
#if ENABLE_ENV_VARS
    : pStride(getenv("WEED_PSTRIDEPOW") ? pow2Gpu((vecLenInt)std::stoi(std::string(getenv("WEED_PSTRIDEPOW"))))
                                         : pow2Gpu((vecLenInt)PSTRIDEPOW))
#else
    : pStride(pow2Gpu((vecLenInt)PSTRIDEPOW))
#endif
#if ENABLE_PTHREAD
    , numCores(std::thread::hardware_concurrency())
#else
    , numCores(1U)
#endif
{
    const vecLenInt pStridePow = log2Gpu(pStride);
    const vecLenInt minStridePow = (numCores > 1U) ? (vecLenInt)pow2Gpu(log2Gpu(numCores - 1U)) : 0U;
    dispatchThreshold = (pStridePow > minStridePow) ? (pStridePow - minStridePow) : 0U;
}

void ParallelFor::par_for(const vecCapIntGpu begin, const vecCapIntGpu end, ParallelFunc fn)
{
    par_for_inc(
        begin, end - begin, [](const vecCapIntGpu& i) { return i; }, fn);
}

void ParallelFor::par_for_set(const std::set<vecCapIntGpu>& sparseSet, ParallelFunc fn)
{
    par_for_inc(
        0U, sparseSet.size(),
        [&sparseSet](const vecCapIntGpu& i) {
            auto it = sparseSet.begin();
            std::advance(it, i);
            return *it;
        },
        fn);
}

void ParallelFor::par_for_set(const std::vector<vecCapIntGpu>& sparseSet, ParallelFunc fn)
{
    par_for_inc(
        0U, sparseSet.size(),
        [&sparseSet](const vecCapIntGpu& i) {
            auto it = sparseSet.begin();
            std::advance(it, i);
            return *it;
        },
        fn);
}

void ParallelFor::par_for_sparse_compose(const std::vector<vecCapIntGpu>& lowSet,
    const std::vector<vecCapIntGpu>& highSet, const vecLenInt& highStart, ParallelFunc fn)
{
    const vecCapIntGpu lowSize = lowSet.size();
    par_for_inc(
        0U, lowSize * highSet.size(),
        [&lowSize, &highStart, &lowSet, &highSet](const vecCapIntGpu& i) {
            const vecCapIntGpu lowPerm = i % lowSize;
            const vecCapIntGpu highPerm = (i - lowPerm) / lowSize;
            auto it = lowSet.begin();
            std::advance(it, lowPerm);
            vecCapIntGpu perm = *it;
            it = highSet.begin();
            std::advance(it, highPerm);
            perm |= (*it) << highStart;
            return perm;
        },
        fn);
}

void ParallelFor::par_for_skip(const vecCapIntGpu begin, const vecCapIntGpu end, const vecCapIntGpu skipMask,
    const vecLenInt maskWidth, ParallelFunc fn)
{
    /*
     * Add maskWidth bits by shifting the incrementor up that number of
     * bits, filling with 0's.
     *
     * For example, if the skipMask is 0x8, then the lowMask will be 0x7
     * and the high mask will be ~(0x7 + 0x8) ==> ~0xf, shifted by the
     * number of extra bits to add.
     */

    if ((skipMask << maskWidth) >= end) {
        // If we're skipping trailing bits, this is much cheaper:
        return par_for(begin, skipMask, fn);
    }

    const vecCapIntGpu lowMask = skipMask - 1U;
    const vecCapIntGpu highMask = ~lowMask;

    IncrementFunc incFn;
    if (!lowMask) {
        // If we're skipping leading bits, this is much cheaper:
        incFn = [maskWidth](const vecCapIntGpu& i) { return (i << maskWidth); };
    } else {
        incFn = [lowMask, highMask, maskWidth](
                    const vecCapIntGpu& i) { return ((i & lowMask) | ((i & highMask) << maskWidth)); };
    }

    par_for_inc(begin, (end - begin) >> maskWidth, incFn, fn);
}

void ParallelFor::par_for_mask(
    const vecCapIntGpu begin, const vecCapIntGpu end, const std::vector<vecCapIntGpu>& maskArray, ParallelFunc fn)
{
    const vecLenInt maskLen = maskArray.size();
    /* Pre-calculate the masks to simplify the increment function later. */
    std::unique_ptr<vecCapIntGpu[][2]> masks(new vecCapIntGpu[maskLen][2]);

    bool onlyLow = true;
    for (vecLenInt i = 0; i < maskLen; ++i) {
        masks[i][0U] = maskArray[i] - 1U; // low mask
        masks[i][1U] = (~(masks[i][0U] + maskArray[i])); // high mask
        if (maskArray[maskLen - i - 1U] != (end >> (i + 1U))) {
            onlyLow = false;
        }
    }

    IncrementFunc incFn;
    if (onlyLow) {
        par_for(begin, end >> maskLen, fn);
    } else {
        incFn = [&masks, maskLen](const vecCapIntGpu& iConst) {
            /* Push i apart, one mask at a time. */
            vecCapIntGpu i = iConst;
            for (vecLenInt m = 0U; m < maskLen; ++m) {
                i = ((i << 1U) & masks[m][1U]) | (i & masks[m][0U]);
            }
            return i;
        };

        par_for_inc(begin, (end - begin) >> maskLen, incFn, fn);
    }
}

#if ENABLE_PTHREAD
/*
 * Iterate through the permutations a maximum of end-begin times, allowing the
 * caller to control the incrementation offset through 'inc'.
 */
void ParallelFor::par_for_inc(
    const vecCapIntGpu begin, const vecCapIntGpu itemCount, IncrementFunc inc, ParallelFunc fn)
{
    const vecCapIntGpu Stride = pStride;
    unsigned threads = (unsigned)(itemCount / pStride);
    if (threads > numCores) {
        threads = numCores;
    }

    if (threads <= 1U) {
        const vecCapIntGpu maxLcv = begin + itemCount;
        for (vecCapIntGpu j = begin; j < maxLcv; ++j) {
            fn(inc(j), 0U);
        }

        return;
    }

    DECLARE_ATOMIC_vecCapInt();
    idx = 0U;
    std::vector<std::future<void>> futures;
    futures.reserve(threads);
    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures.emplace_back(ATOMIC_ASYNC(cpu, &idx, &begin, &itemCount, &Stride, inc, fn) {
            for (;;) {
                vecCapIntGpu i;
                ATOMIC_INC();
                const vecCapIntGpu l = i * Stride;
                if (l >= itemCount) {
                    break;
                }
                const vecCapIntGpu maxJ = ((l + Stride) < itemCount) ? Stride : (itemCount - l);
                for (vecCapIntGpu j = 0U; j < maxJ; ++j) {
                    fn(inc(begin + j + l), cpu);
                }
            }
        }));
    }

    for (std::future<void>& future : futures) {
        future.get();
    }
}

real1_f ParallelFor::par_norm(const vecCapIntGpu itemCount, const StateVectorPtr stateArray, real1_f norm_thresh)
{
    if (norm_thresh <= ZERO_R1) {
        return par_norm_exact(itemCount, stateArray);
    }

    const vecCapIntGpu Stride = pStride;
    unsigned threads = (unsigned)(itemCount / pStride);
    if (threads > numCores) {
        threads = numCores;
    }
    if (threads <= 1U) {
        real1 nrmSqr = ZERO_R1;
        const real1 nrm_thresh = (real1)norm_thresh;
        for (vecCapIntGpu j = 0U; j < itemCount; ++j) {
            const real1 nrm = norm(stateArray->read(j));
            if (nrm >= nrm_thresh) {
                nrmSqr += nrm;
            }
        }

        return (real1_f)nrmSqr;
    }

    DECLARE_ATOMIC_vecCapInt();
    idx = 0U;
    std::vector<std::future<real1_f>> futures;
    futures.reserve(threads);
    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures.emplace_back(ATOMIC_ASYNC(&idx, &itemCount, stateArray, &Stride, &norm_thresh) {
            const real1 nrm_thresh = (real1)norm_thresh;
            real1 sqrNorm = ZERO_R1;
            for (;;) {
                vecCapIntGpu i;
                ATOMIC_INC();
                const vecCapIntGpu l = i * Stride;
                if (l >= itemCount) {
                    break;
                }
                const vecCapIntGpu maxJ = ((l + Stride) < itemCount) ? Stride : (itemCount - l);
                for (vecCapIntGpu j = 0U; j < maxJ; ++j) {
                    vecCapIntGpu k = i * Stride + j;
                    const real1 nrm = norm(stateArray->read(k));
                    if (nrm >= nrm_thresh) {
                        sqrNorm += nrm;
                    }
                }
            }
            return (real1_f)sqrNorm;
        }));
    }

    real1_f nrmSqr = ZERO_R1_F;
    for (std::future<real1_f>& future : futures) {
        nrmSqr += future.get();
    }

    return nrmSqr;
}

real1_f ParallelFor::par_norm_exact(const vecCapIntGpu itemCount, const StateVectorPtr stateArray)
{
    const vecCapIntGpu Stride = pStride;
    unsigned threads = (unsigned)(itemCount / pStride);
    if (threads > numCores) {
        threads = numCores;
    }
    if (threads <= 1U) {
        real1 nrmSqr = ZERO_R1;
        for (vecCapIntGpu j = 0U; j < itemCount; ++j) {
            nrmSqr += norm(stateArray->read(j));
        }

        return (real1_f)nrmSqr;
    }

    DECLARE_ATOMIC_vecCapInt();
    idx = 0U;
    std::vector<std::future<real1_f>> futures;
    futures.reserve(threads);
    for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures.emplace_back(ATOMIC_ASYNC(&idx, &itemCount, &Stride, stateArray) {
            real1 sqrNorm = ZERO_R1;
            for (;;) {
                vecCapIntGpu i;
                ATOMIC_INC();
                const vecCapIntGpu l = i * Stride;
                if (l >= itemCount) {
                    break;
                }
                const vecCapIntGpu maxJ = ((l + Stride) < itemCount) ? Stride : (itemCount - l);
                for (vecCapIntGpu j = 0U; j < maxJ; ++j) {
                    sqrNorm += norm(stateArray->read(i * Stride + j));
                }
            }
            return (real1_f)sqrNorm;
        }));
    }

    real1_f nrmSqr = ZERO_R1_F;
    for (std::future<real1_f>& future : futures) {
        nrmSqr += future.get();
    }

    return nrmSqr;
}
#else
/*
 * Iterate through the permutations a maximum of end-begin times, allowing the
 * caller to control the incrementation offset through 'inc'.
 */
void ParallelFor::par_for_inc(
    const vecCapIntGpu begin, const vecCapIntGpu itemCount, IncrementFunc inc, ParallelFunc fn)
{
    const vecCapIntGpu maxLcv = begin + itemCount;
    for (vecCapIntGpu j = begin; j < maxLcv; ++j) {
        fn(inc(j), 0U);
    }
}

real1_f ParallelFor::par_norm(const vecCapIntGpu itemCount, const StateVectorPtr stateArray, real1_f norm_thresh)
{
    if (norm_thresh <= ZERO_R1) {
        return par_norm_exact(itemCount, stateArray);
    }

    real1_f nrmSqr = ZERO_R1;
    for (vecCapIntGpu j = 0U; j < itemCount; ++j) {
        const real1_f nrm = norm(stateArray->read(j));
        if (nrm >= norm_thresh) {
            nrmSqr += nrm;
        }
    }

    return nrmSqr;
}

real1_f ParallelFor::par_norm_exact(const vecCapIntGpu itemCount, const StateVectorPtr stateArray)
{
    real1_f nrmSqr = ZERO_R1;
    for (vecCapIntGpu j = 0U; j < itemCount; ++j) {
        nrmSqr += norm(stateArray->read(j));
    }

    return nrmSqr;
}
#endif
} // namespace Weed
