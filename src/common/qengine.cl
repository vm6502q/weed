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

// "zdiv" contributed by Elara, an OpenAI custom GPT
inline cmplx zdiv(const cmplx lhs, const cmplx rhs)
{
    const real1 inv = (real1)1 / ((rhs.x * rhs.x) + (rhs.y * rhs.y));

    return (cmplx)(
        ((lhs.x * rhs.x) + (lhs.y * rhs.y)) * inv,
        ((lhs.y * rhs.x) - (lhs.x * rhs.y)) * inv
    );
}

inline cmplx2 zmatrixmul(const real1 nrm, const cmplx4 lhs, const cmplx2 rhs)
{
    return nrm *
        ((cmplx2)((lhs.lo.x * rhs.x) - (lhs.lo.y * rhs.y) + (lhs.lo.z * rhs.z) - (lhs.lo.w * rhs.w),
            (lhs.lo.x * rhs.y) + (lhs.lo.y * rhs.x) + (lhs.lo.z * rhs.w) + (lhs.lo.w * rhs.z),
            (lhs.hi.x * rhs.x) - (lhs.hi.y * rhs.y) + (lhs.hi.z * rhs.z) - (lhs.hi.w * rhs.w),
            (lhs.hi.x * rhs.y) + (lhs.hi.y * rhs.x) + (lhs.hi.z * rhs.w) + (lhs.hi.w * rhs.z)));
}

inline cmplx zpow_real(const cmplx z, const real1 p)
{
    const real1 r = hypot(z.x, z.y);
    const real1 theta = atan2(z.y, z.x);

    const real1 rp = pow(r, p);
    const real1 pt = p * theta;

    return rp * sin((cmplx)(pt + SineShift, pt));
}

inline cmplx zexp(const cmplx z)
{
    return exp(z.x) * sin((cmplx)(z.y + SineShift, z.y));
}

inline cmplx zlog(const cmplx z)
{
    const real1 r = hypot(z.x, z.y);
    const real1 theta = atan2(z.y, z.x);

    return (cmplx)(log(r), theta);
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
#define O_A vecCapIntArgs[0U]
#define I_A vecCapIntArgs[1U]
#define O_B vecCapIntArgs[2U]
#define I_B vecCapIntArgs[3U]
#define O_C vecCapIntArgs[4U]
#define I_C vecCapIntArgs[5U]
#define J_A vecCapIntArgs[6U]
#define J_B vecCapIntArgs[7U]
#define J_C vecCapIntArgs[8U]
#define K   vecCapIntArgs[9U]
#define l_X get_group_id(0)
#define l_Y get_group_id(1)
#define l_Z get_group_id(2)

#define SUM_LOCAL(v, l_buffer, g_buffer)                                                                               \
    const vecCapIntGpu locID = get_local_id(0);                                                                        \
    const vecCapIntGpu locNthreads = get_local_size(0);                                                                \
    l_buffer[locID] = v[i_X * I_A + O_A];                                                                              \
                                                                                                                       \
    for (vecCapIntGpu lcv = (locNthreads >> 1U); lcv > 0U; lcv >>= 1U) {                                               \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                                  \
        if (locID < lcv) {                                                                                             \
            l_buffer[locID] += l_buffer[locID + lcv];                                                                  \
        }                                                                                                              \
    }                                                                                                                  \
                                                                                                                       \
    if (locID == 0U) {                                                                                                 \
        g_buffer[l_X * I_A] = l_buffer[0U];                                                                            \
    }

void kernel clear_buffer_real(global real1* a)
{
    a[i_X] = ZERO_R1;
}
void kernel fill_ones_real(global real1* a)
{
    a[i_X] = ONE_R1;
}
void kernel fill_ones_complex(global cmplx* a)
{
    a[i_X] = (cmplx)(ONE_R1, ZERO_R1);
}
void kernel fill_value_real(global real1* a, constant real1* v)
{
    a[i_X] = *v;
}
void kernel fill_value_complex(global cmplx* a, constant cmplx* v)
{
    a[i_X] = *v;
}
void kernel real_to_complex_buffer(global real1* a, global cmplx* b)
{
    b[i_X] = (cmplx)(a[i_X], ZERO_R1);
}

void kernel reduce_real(global real1* a, global real1* b, constant vecCapIntGpu* shape, constant vecCapIntGpu* stride, constant vecCapIntGpu* vecCapIntArgs)
{
  vecCapIntGpu tmp = i_X;
  vecCapIntGpu base = O_A;
  const vecCapIntGpu id = I_A;

  for (int d = O_C - 1; d >= 0; --d) {
    if (d == id) {
      continue;
    }

    vecCapIntGpu dim = shape[d];
    vecCapIntGpu i_d = tmp % dim;
    tmp /= dim;

    base += i_d * stride[d];
  }

  real1 sum = ZERO_R1;
  for (vecCapIntGpu j = 0U; j < shape[id]; ++j) {
    sum += a[base + j * stride[id]];
  }
  b[i_X * I_B] = sum;
}

void kernel reduce_complex(global cmplx* a, global cmplx* b, constant vecCapIntGpu* shape, constant vecCapIntGpu* stride, constant vecCapIntGpu* vecCapIntArgs)
{
  vecCapIntGpu tmp = i_X;
  vecCapIntGpu base = O_A;
  const vecCapIntGpu id = I_A;

  for (int d = O_C - 1; d >= 0; --d) {
    if (d == id) {
      continue;
    }

    vecCapIntGpu dim = shape[d];
    vecCapIntGpu i_d = tmp % dim;
    tmp /= dim;

    base += i_d * stride[d];
  }

  cmplx sum = ZERO_R1;
  for (vecCapIntGpu j = 0U; j < shape[id]; ++j) {
    sum += a[base + j * stride[id]];
  }
  b[i_X * I_B] = sum;
}

void kernel relu(global real1* a, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    const real1 tmp = a[i_X * I_A + O_A];
    out[i_X * I_B] = (tmp > 0) ? tmp : ZERO_R1;
}
void kernel relu_grad_real(global real1* din, global real1* in, global real1* dout, constant vecCapIntGpu* vecCapIntArgs)
{
    din[i_X * I_A + O_A] += (in[i_X * I_B + O_B] > 0) ? dout[i_X * I_C + O_C] : ZERO_R1;
}
void kernel relu_grad_complex(global cmplx* din, global real1* in, global cmplx* dout, constant vecCapIntGpu* vecCapIntArgs)
{
    din[i_X * I_A + O_A] += (in[i_X * I_B + O_B] > 0) ? dout[i_X * I_C + O_C] : (cmplx)(ZERO_R1, ZERO_R1);
}

void kernel sigmoid(global real1* a, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_B] = ONE_R1 / (ONE_R1 + exp(-a[i_X * I_A + O_A]));
}
void kernel sigmoid_grad_real(global real1* din, global real1* in, global real1* dout, constant vecCapIntGpu* vecCapIntArgs)
{
    const real1 yi = in[i_X * I_B + O_B];
    din[i_X * I_A + O_A] += dout[i_X * I_C + O_C] * yi * (ONE_R1 - yi);
}
void kernel sigmoid_grad_complex(global cmplx* din, global real1* in, global cmplx* dout, constant vecCapIntGpu* vecCapIntArgs)
{
    const cmplx yi = in[i_X * I_B + O_B];
    din[i_X * I_A + O_A] += zmul(zmul(dout[i_X * I_C + O_C], yi), ((cmplx)(ONE_R1, ZERO_R1) - yi));
}

void kernel abs_real(global real1* a, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    const real1 tmp = a[i_X * I_A + O_A];
    out[i_X * I_B] = (tmp < 0) ? -tmp : tmp;
}
void kernel abs_complex(global cmplx* a, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    const cmplx tmp = a[i_X * I_A + O_A];
    out[i_X * I_B] = sqrt(dot(tmp, tmp));
}
void kernel abs_real_grad_real(global real1* din, global real1* in, global real1* dout, constant vecCapIntGpu* vecCapIntArgs)
{
    const real1 tmp = in[i_X * I_B + O_B];
    const real1 tmp_o = dout[i_X * I_C + O_C];
    din[i_X * I_A + O_A] += (tmp == ZERO_R1) ? ZERO_R1 : ((tmp > ZERO_R1) ? tmp_o : -tmp_o);
}
void kernel abs_real_grad_complex(global cmplx* din, global real1* in, global cmplx* dout, constant vecCapIntGpu* vecCapIntArgs)
{
    const real1 tmp = in[i_X * I_B + O_B];
    const cmplx tmp_o = dout[i_X * I_C + O_C];
    din[i_X * I_A + O_A] += (tmp == ZERO_R1) ? (cmplx)(ZERO_R1, ZERO_R1) : ((tmp > ZERO_R1) ? tmp_o : -tmp_o);
}
void kernel abs_complex_grad_real(global cmplx* din, global cmplx* in, global real1* dout, constant vecCapIntGpu* vecCapIntArgs)
{
    const cmplx ZERO_CMPLX = (cmplx)(ZERO_R1, ZERO_R1);
    const cmplx tmp = in[i_X * I_B + O_B];
    const real1 tmp_o = dout[i_X * I_C + O_C];
    const real1 out = sqrt(dot(tmp, tmp));
    din[i_X * I_A + O_A] += (tmp == ZERO_CMPLX) ? ZERO_CMPLX : ((tmp_o / out) * tmp);
}
void kernel abs_complex_grad_complex(global cmplx* din, global cmplx* in, global cmplx* dout, constant vecCapIntGpu* vecCapIntArgs)
{
    const cmplx ZERO_CMPLX = (cmplx)(ZERO_R1, ZERO_R1);
    const cmplx tmp = in[i_X * I_B + O_B];
    const cmplx tmp_o = dout[i_X * I_C + O_C];
    const real1 out = sqrt(dot(tmp, tmp));
    din[i_X * I_A + O_A] += (tmp == ZERO_CMPLX) ? ZERO_CMPLX : (zmul(tmp_o, tmp) / out);
}

void kernel add_real(global real1* a, global real1* b, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C] = a[i_X * I_A + O_A] + b[i_X * I_B + O_B];
}
void kernel add_complex(global cmplx* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C] = a[i_X * I_A + O_A] + b[i_X * I_B + O_B];
}
void kernel add_mixed(global cmplx* a, global real1* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C] = a[i_X * I_A + O_A] + (cmplx)(b[i_X * I_B + O_B], 0);
}

void kernel mul_real(global real1* a, global real1* b, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C] = a[i_X * I_A + O_A] * b[i_X * I_B + O_B];
}
void kernel mul_complex(global cmplx* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C] = zmul(a[i_X * I_A + O_A], b[i_X * I_B + O_B]);
}
void kernel mul_mixed(global cmplx* a, global real1* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C] = b[i_X * I_B + O_B] * a[i_X * I_A + O_A];
}

void kernel matmul_real(global real1* a, global real1* b, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    real1 sum = ZERO_R1;
    for (vecCapIntGpu k = 0; k < K; ++k) {
        const vecCapIntGpu a_idx = (O_A + i_X * I_A + k * J_A);
        const vecCapIntGpu b_idx = (O_B + k * I_B + i_Y * J_B);
        sum += a[a_idx] * b[b_idx];
    }
    const vecCapIntGpu o_idx = i_X * I_C + i_Y * J_C;
    out[o_idx] = sum;
}
void kernel matmul_complex(global cmplx* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    cmplx sum = ZERO_R1;
    for (vecCapIntGpu k = 0; k < K; ++k) {
        const vecCapIntGpu a_idx = (O_A + i_X * I_A + k * J_A);
        const vecCapIntGpu b_idx = (O_B + k * I_B + i_Y * J_B);
        sum += zmul(a[a_idx], b[b_idx]);
    }
    const vecCapIntGpu o_idx = i_X * I_C + i_Y * J_C;
    out[o_idx] = sum;
}
void kernel matmul_mixed_c_left(global cmplx* a, global real1* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    cmplx sum = ZERO_R1;
    for (vecCapIntGpu k = 0; k < K; ++k) {
        const vecCapIntGpu a_idx = (O_A + i_X * I_A + k * J_A);
        const vecCapIntGpu b_idx = (O_B + k * I_B + i_Y * J_B);
        sum += b[b_idx] * a[a_idx];
    }
    const vecCapIntGpu o_idx = i_X * I_C + i_Y * J_C;
    out[o_idx] = sum;
}
void kernel matmul_mixed_c_right(global real1* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    cmplx sum = ZERO_R1;
    for (vecCapIntGpu k = 0; k < K; ++k) {
        const vecCapIntGpu a_idx = (O_A + i_X * I_A + k * J_A);
        const vecCapIntGpu b_idx = (O_B + k * I_B + i_Y * J_B);
        sum += a[a_idx] * b[b_idx];
    }
    const vecCapIntGpu o_idx = i_X * I_C + i_Y * J_C;
    out[o_idx] = sum;
}

void kernel sub_real(global real1* a, global real1* b, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C] = a[i_X * I_A + O_A] - b[i_X * I_B + O_B];
}
void kernel sub_complex(global cmplx* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C] = a[i_X * I_A + O_A] - b[i_X * I_B + O_B];
}
void kernel sub_mixed_c_left(global cmplx* a, global real1* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C] = a[i_X * I_A + O_A] - (cmplx)(b[i_X * I_B + O_B], ZERO_R1);
}
void kernel sub_mixed_c_right(global real1* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C] = (cmplx)(a[i_X * I_A + O_A], ZERO_R1) - b[i_X * I_B + O_B];
}

void kernel div_real(global real1* a, global real1* b, global real1* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C] = a[i_X * I_A + O_A] / b[i_X * I_B + O_B];
}
void kernel div_complex(global cmplx* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C] = zdiv(a[i_X * I_A + O_A], b[i_X * I_B + O_B]);
}
void kernel div_mixed_c_left(global cmplx* a, global real1* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C] = a[i_X * I_A + O_A] / b[i_X * I_B + O_B];
}
void kernel div_mixed_c_right(global real1* a, global cmplx* b, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs)
{
    out[i_X * I_C] = zdiv((cmplx)(a[i_X * I_A + O_A], ZERO_R1), b[i_X * I_B + O_B]);
}

void kernel add_in_place_real(global real1* a, global real1* b, constant vecCapIntGpu* vecCapIntArgs)
{
    a[i_X * I_A + O_A] += b[i_X * I_B + O_B];
}
void kernel add_in_place_complex(global cmplx* a, global cmplx* b, constant vecCapIntGpu* vecCapIntArgs)
{
    a[i_X * I_A + O_A] += b[i_X * I_B + O_B];
}
void kernel add_in_place_mixed(global cmplx* a, global real1* b, constant vecCapIntGpu* vecCapIntArgs)
{
    a[i_X * I_A + O_A] += (cmplx)(b[i_X * I_B + O_B], 0);
}

void kernel sub_in_place_real(global real1* a, global real1* b, constant vecCapIntGpu* vecCapIntArgs)
{
    a[i_X * I_A + O_A] -= b[i_X * I_B + O_B];
}
void kernel sub_in_place_complex(global cmplx* a, global cmplx* b, constant vecCapIntGpu* vecCapIntArgs)
{
    a[i_X * I_A + O_A] -= b[i_X * I_B + O_B];
}
void kernel sub_in_place_mixed(global cmplx* a, global real1* b, constant vecCapIntGpu* vecCapIntArgs)
{
    a[i_X * I_A + O_A] -= (cmplx)(b[i_X * I_B + O_B], 0);
}

void kernel pow_real(global real1* a, global real1* out, constant vecCapIntGpu* vecCapIntArgs, constant real1* p)
{
    out[i_X * I_B] = pow(a[i_X * I_A + O_A], *p);
}
void kernel pow_complex(global cmplx* a, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs, constant real1* p)
{
    out[i_X * I_B] = zpow_real(a[i_X * I_A + O_A], *p);
}
void kernel exp_real(global real1* a, global real1* out, constant vecCapIntGpu* vecCapIntArgs, constant real1* log_b)
{
    out[i_X * I_B] = exp(a[i_X * I_A + O_A] * (*log_b));
}
void kernel exp_complex(global cmplx* a, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs, constant real1* log_b)
{
    out[i_X * I_B] = zexp(a[i_X * I_A + O_A] * (*log_b));
}
void kernel log_real(global real1* a, global real1* out, constant vecCapIntGpu* vecCapIntArgs, constant real1* inv_log_b)
{
    out[i_X * I_B] = log(a[i_X * I_A + O_A]) * (*inv_log_b);
}
void kernel log_complex(global cmplx* a, global cmplx* out, constant vecCapIntGpu* vecCapIntArgs, constant real1* inv_log_b)
{
    out[i_X * I_B] = zlog(a[i_X * I_A + O_A]) * (*inv_log_b);
}

void kernel clamp_real(global real1* a, global real1* out, constant vecCapIntGpu* vecCapIntArgs, constant cmplx* p)
{
    const real1 tmp = a[i_X * I_A + O_A];
    const cmplx b = *p;
    out[i_X * I_B] = (tmp > b.x) ? ((tmp < b.y) ? tmp  : b.y) : b.x;
}
void kernel clamp_grad_real(global real1* dy, global real1* x, global real1* dx, constant vecCapIntGpu* vecCapIntArgs, constant cmplx* p)
{
    const real1 xi = x[i_X * I_B + O_B];
    const cmplx b = *p;
    if (xi > b.x && xi < b.y) {
      dx[i_X * I_C + O_C] += dy[i_X * I_A + O_A];
    }
}
void kernel clamp_grad_complex(global cmplx* dy, global real1* x, global cmplx* dx, constant vecCapIntGpu* vecCapIntArgs, constant cmplx* p)
{
    const real1 xi = x[i_X * I_B + O_B];
    const cmplx b = *p;
    if (xi > b.x && xi < b.y) {
      dx[i_X * I_C + O_C] += dy[i_X * I_A + O_A];
    }
}
