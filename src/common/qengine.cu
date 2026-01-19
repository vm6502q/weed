//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "common/cuda_kernels.cuh"

namespace Weed {

__device__ inline qCudaCmplx zmul(const qCudaCmplx lhs, const qCudaCmplx rhs)
{
    return make_qCudaCmplx((lhs.x * rhs.x) - (lhs.y * rhs.y), (lhs.x * rhs.y) + (lhs.y * rhs.x));
}

#if FPPOW > 4
__device__ inline qCudaCmplx2 zmatrixmul(const qCudaReal1 nrm, const qCudaReal1* lhs, const qCudaCmplx2 rhs)
{
    return (make_qCudaCmplx2(nrm * ((lhs[0] * rhs.x) - (lhs[1] * rhs.y) + (lhs[2] * rhs.z) - (lhs[3] * rhs.w)),
        nrm * ((lhs[0] * rhs.y) + (lhs[1] * rhs.x) + (lhs[2] * rhs.w) + (lhs[3] * rhs.z)),
        nrm * ((lhs[4] * rhs.x) - (lhs[5] * rhs.y) + (lhs[6] * rhs.z) - (lhs[7] * rhs.w)),
        nrm * ((lhs[4] * rhs.y) + (lhs[5] * rhs.x) + (lhs[6] * rhs.w) + (lhs[7] * rhs.z))));
}
#else
__device__ inline void zmatrixmul(const qCudaReal1 nrm, const qCudaReal1* lhs, const qCudaCmplx2 rhs, qCudaCmplx2 o)
{
    o[0] =
        (make_qCudaCmplx(nrm * ((lhs[0] * rhs[0].x) - (lhs[1] * rhs[0].y) + (lhs[2] * rhs[1].x) - (lhs[3] * rhs[1].y)),
            nrm * ((lhs[0] * rhs[0].y) + (lhs[1] * rhs[0].x) + (lhs[2] * rhs[1].y) + (lhs[3] * rhs[1].x))));
    o[1] =
        (make_qCudaCmplx(nrm * ((lhs[4] * rhs[0].x) - (lhs[5] * rhs[0].y) + (lhs[6] * rhs[1].x) - (lhs[7] * rhs[1].y)),
            nrm * ((lhs[4] * rhs[0].y) + (lhs[5] * rhs[0].x) + (lhs[6] * rhs[1].y) + (lhs[7] * rhs[1].x))));
}
#endif

__device__ inline qCudaReal1 qCudaArg(const qCudaCmplx cmp)
{
    if (cmp.x == (qCudaReal1)0.0f && cmp.y == (qCudaReal1)0.0f)
        return (qCudaReal1)0.0f;
    return (qCudaReal1)atan2((qCudaReal1_f)cmp.y, (qCudaReal1_f)cmp.x);
}

__device__ inline qCudaReal1 qCudaDot(qCudaReal2 a, qCudaReal2 b) { return a.x * b.x + a.y * b.y; }

__device__ inline qCudaReal1 qCudaDot(qCudaReal4 a, qCudaReal4 b)
{
#if FPPOW > 4
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
#else
    return a[0].x * b[0].x + a[0].y * b[0].y + a[1].x * b[1].x + a[1].y * b[1].y;
#endif
}

#if FPPOW > 4
__device__ inline qCudaCmplx polar_unit(const qCudaReal1 theta) { return make_qCudaCmplx(cos(theta), sin(theta)); }
#else
__device__ inline qCudaCmplx polar_unit(const qCudaReal1 theta)
{
    return make_qCudaCmplx((qCudaReal1)cos((qCudaReal1_f)theta), (qCudaReal1)sin((qCudaReal1_f)theta));
}
#endif

__device__ inline qCudaCmplx qCudaConj(qCudaCmplx a) { return make_qCudaCmplx(a.x, -a.y); }
} // namespace Weed
