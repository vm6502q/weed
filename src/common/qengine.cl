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

#define i_X get_global_id(0)
#define i_Y get_global_id(1)
#define i_Z get_global_id(2)
#define S_X get_global_size(0)
#define S_Y get_global_size(1)
#define S_Z get_global_size(2)
#define O_A vecCapIntArgs[0]
#define I_A vecCapIntArgs[1]
#define O_B vecCapIntArgs[2]
#define I_B vecCapIntArgs[3]
#define O_C vecCapIntArgs[4]
#define I_C vecCapIntArgs[5]
#define S_K vecCapIntArgs[6]

void kernel add_real(global real1* a, global real1* b, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = a[i_X * I_A + O_A] + b[i_X * I_B + O_B];
}
void kernel add_complex(global cmplx* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = a[i_X * I_A + O_A] + b[i_X * I_B + O_B];
}
void kernel add_mixed(global cmplx* a, global real1* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = a[i_X * I_A + O_A] + (cmplx)(b[i_X * I_B + O_B], 0);
}

void kernel add_real_inplace(global real1* a, global real1* b, constant vecCapIntGpu* vecCapIntArgs)
{
    a[i_X * I_A + O_A] +=  b[i_X * I_B + O_B];
}
void kernel add_complex_inplace(global cmplx* a, global cmplx* b, constant vecCapIntGpu* vecCapIntArgs)
{
    a[i_X * I_A + O_A] += b[i_X * I_B + O_B];
}
void kernel add_mixed_inplace(global cmplx* a, global real1* b, constant vecCapIntGpu* vecCapIntArgs)
{
    a[i_X * I_A + O_A] += (cmplx)(b[i_X * I_B + O_B], 0);
}

void kernel mul_real(global real1* a, global real1* b, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = a[i_X * I_A + O_A] * b[i_X * I_B + O_B];
}
void kernel mul_complex(global cmplx* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = zmul(a[i_X * I_A + O_A], b[i_X * I_B + O_B]);
}
void kernel mul_mixed(global cmplx* a, global real1* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = b[i_X * I_B + O_B] * a[i_X * I_A + O_A];
}

void kernel mul_real_inplace(global real1* a, global real1* b, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    a[i_X * I_A + O_A] *= b[i_X * I_B + O_B];
}
void kernel mul_complex_inplace(global cmplx* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    a[i_X * I_A + O_A] = zmul(a[i_X * I_A + O_A], b[i_X * I_B + O_B]);
}
void kernel mul_mixed_inplace(global cmplx* a, global real1* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    a[i_X * I_A + O_A] = b[i_X * I_B + O_B] * a[i_X * I_A + O_A];
}

void kernel relu(global real1* a, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    const real1 tmp = a[i_X * I_A + O_A];
    out[i_X * I_B + O_B] = (tmp > 0) ? tmp : ZERO_R1;
}
void kernel relu_grad(global real1* din, global real1* in, global real1* dout, constant vecCapIntGpu* vecCapIntArgs)
{
    din[i_X * I_A + O_A] = (in[i_X * I_B + O_B] > 0) ? dout[i_X * I_C + O_C] : ZERO_R1;
}

void kernel matmul_real(global real1* a, global real1* b, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    real1 sum = ZERO_R1;
    for (vecCapIntGpu k = 0; k < S_K; ++k) {
        sum += a[(i_X * S_K + k) * I_A + O_A] * b[(k * S_Y + i_Y) * I_B + O_B];
    }
    out[(i_X * S_Y + i_Y) * I_C + O_C] = sum;
}
void kernel matmul_complex(global cmplx* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    cmplx sum = ZERO_R1;
    for (vecCapIntGpu k = 0; k < S_K; ++k) {
        sum += zmul(a[(i_X * S_K + k) * I_A + I_A], b[(k * S_Y + i_Y) * I_B + O_B]);
    }
    out[(i_X * S_Y + i_Y) * I_C + O_C] = sum;
}
void kernel matmul_mixed_c_left(global cmplx* a, global real1* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    cmplx sum = ZERO_R1;
    for (vecCapIntGpu k = 0; k < S_K; ++k) {
        sum += b[(k * S_Y + i_Y) * I_B + O_B] * a[(i_X * S_K + k) * I_A + O_A];
    }
    out[(i_X * S_Y + i_Y) * I_C + O_C] = sum;
}
void kernel matmul_mixed_c_right(global real1* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    cmplx sum = ZERO_R1;
    for (vecCapIntGpu k = 0; k < S_K; ++k) {
        sum += a[(i_X * S_K + k) * I_A + I_A] * b[(k * S_Y + i_Y) * I_B + I_B];
    }
    out[(i_X * S_Y + i_Y) * I_C + O_C] = sum;
}
