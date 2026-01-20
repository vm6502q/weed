//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "storage.hpp"

namespace Weed {
struct CpuRealStorage : Storage {
    Real1Ptr data;

    CpuRealStorage(vecCapIntGpu n)
        : data(Alloc(n))
    {
        device = DeviceTag::CPU;
        dtype = DType::REAL;
        size = n;
    }

    ~CpuRealStorage() {}

#if defined(__APPLE__)
    static real1* _aligned_state_vec_alloc(vecCapIntGpu allocSize)
    {
        void* toRet;
        posix_memalign(&toRet, WEED_ALIGN_SIZE, allocSize);
        return (real1*)toRet;
    }
#endif

    static std::unique_ptr<real1[], void (*)(real1*)> Alloc(vecCapIntGpu elemCount)
    {
#if defined(__ANDROID__)
        return std::unique_ptr<real1[], void (*)(real1*)>(new real1[elemCount], [](real1* c) { delete c; });
#else
        // elemCount is always a power of two, but might be smaller than WEED_ALIGN_SIZE
        size_t allocSize = sizeof(real1) * elemCount;
        if (allocSize < WEED_ALIGN_SIZE) {
            allocSize = WEED_ALIGN_SIZE;
        }
#if defined(__APPLE__)
        return std::unique_ptr<real1[], void (*)(real1*)>(
            _aligned_state_vec_alloc(allocSize), [](real1* c) { free(c); });
#elif defined(_WIN32) && !defined(__CYGWIN__)
        return std::unique_ptr<real1[], void (*)(real1*)>(
            (real1*)_aligned_malloc(allocSize, WEED_ALIGN_SIZE), [](real1* c) { _aligned_free(c); });
#else
        return std::unique_ptr<real1[], void (*)(real1*)>(
            (real1*)aligned_alloc(WEED_ALIGN_SIZE, allocSize), [](real1* c) { free(c); });
#endif
#endif
    }
};
} // namespace Weed
