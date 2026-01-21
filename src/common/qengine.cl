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

inline cmplx zmul(const cmplx lhs, const cmplx rhs)
{
    return (cmplx)((lhs.x * rhs.x) - (lhs.y * rhs.y), (lhs.x * rhs.y) + (lhs.y * rhs.x));
}

inline cmplx2 zmatrixmul(const real1 nrm, const cmplx4 lhs, const cmplx2 rhs)
{
    return nrm *
        ((cmplx2)((lhs.lo.x * rhs.x) - (lhs.lo.y * rhs.y) + (lhs.lo.z * rhs.z) - (lhs.lo.w * rhs.w),
            (lhs.lo.x * rhs.y) + (lhs.lo.y * rhs.x) + (lhs.lo.z * rhs.w) + (lhs.lo.w * rhs.z),
            (lhs.hi.x * rhs.x) - (lhs.hi.y * rhs.y) + (lhs.hi.z * rhs.z) - (lhs.hi.w * rhs.w),
            (lhs.hi.x * rhs.y) + (lhs.hi.y * rhs.x) + (lhs.hi.z * rhs.w) + (lhs.hi.w * rhs.z)));
}

inline real1 arg(const cmplx cmp)
{
    if (cmp.x == ZERO_R1 && cmp.y == ZERO_R1)
        return ZERO_R1;
    return (real1)atan2((real1_f)cmp.y, (real1_f)cmp.x);
}

inline cmplx conj(const cmplx cmp)
{
    return (cmplx)(cmp.x, -cmp.y);
}

inline cmplx polar_unit(const real1 theta) {
    return sin((cmplx)(theta + (PI_R1 / 2), theta));
}

#define ID get_global_id(0)
#define OFFSET_A vecCapIntArgs[0]
#define OFFSET_B vecCapIntArgs[1]
#define OFFSET_OUT vecCapIntArgs[2]

void kernel add_real(global real1* a, global real1* b, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[ID + OFFSET_OUT] = a[ID + OFFSET_A] + b[ID + OFFSET_B];
}
void kernel add_complex(global cmplx* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[ID + OFFSET_OUT] = a[ID + OFFSET_A] + b[ID + OFFSET_B];
}
void kernel add_mixed(global cmplx* a, global real1* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[ID + OFFSET_OUT] = a[ID + OFFSET_A] + (cmplx)(b[ID + OFFSET_B], 0);
}

void kernel add_real_inplace(global real1* a, global real1* b, constant vecCapIntGpu* vecCapIntArgs)
{
    a[ID + OFFSET_A] +=  b[ID + OFFSET_B];
}
void kernel add_complex_inplace(global cmplx* a, global cmplx* b, constant vecCapIntGpu* vecCapIntArgs)
{
    a[ID + OFFSET_A] += b[ID + OFFSET_B];
}
void kernel add_mixed_inplace(global cmplx* a, global real1* b, constant vecCapIntGpu* vecCapIntArgs)
{
    a[ID + OFFSET_A] += (cmplx)(b[ID + OFFSET_B], 0);
}

void kernel mul_real(global real1* a, global real1* b, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[ID + OFFSET_OUT] = a[ID + OFFSET_A] * b[ID + OFFSET_B];
}
void kernel mul_complex(global cmplx* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[ID + OFFSET_OUT] = zmul(a[ID + OFFSET_A], b[ID + OFFSET_B]);
}
void kernel mul_mixed(global cmplx* a, global real1* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[ID + OFFSET_OUT] = b[ID + OFFSET_B] * a[ID + OFFSET_A];
}

void kernel mul_real_inplace(global real1* a, global real1* b, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    a[ID + OFFSET_A] *= b[ID + OFFSET_B];
}
void kernel mul_complex_inplace(global cmplx* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    a[ID + OFFSET_A] = zmul(a[ID + OFFSET_A], b[ID + OFFSET_B]);
}
void kernel mul_mixed_inplace(global cmplx* a, global real1* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    a[ID + OFFSET_A] = b[ID + OFFSET_B] * a[ID + OFFSET_A];
}

void kernel relu(global real1* a, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    const real1 tmp = a[ID + OFFSET_A];
    out[ID + OFFSET_B] = (tmp > 0) ? tmp : ZERO_R1;
}
