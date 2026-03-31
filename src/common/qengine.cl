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
    const real1_f r = hypot((real1_f)z.x, (real1_f)z.y);
    const real1_f theta = atan2((real1_f)z.y, (real1_f)z.x);

    const real1_f rp = pow(r, (real1_f)p);
    const real1_f pt = p * theta;

    return ((real1)rp) * sin((cmplx)((real1)(pt + SineShift), (real1)pt));
}

inline cmplx zexp(const cmplx z)
{
    return ((real1)exp((real1_f)z.x)) * sin((cmplx)(z.y + SineShift, z.y));
}

inline cmplx zlog(const cmplx z)
{
    const real1_f r = hypot((real1_f)z.x, (real1_f)z.y);
    const real1_f theta = atan2((real1_f)z.y, (real1_f)z.x);

    return (cmplx)((real1)log(r), (real1)theta);
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
#define M   vecCapIntArgs[10]
#define N   vecCapIntArgs[11]
#define l_X get_group_id(0)
#define l_Y get_group_id(1)
#define l_Z get_group_id(2)

void kernel clear_buffer_int(global symint* a)
{
    a[i_X] = 0;
}
void kernel clear_buffer_real(global real1* a)
{
    a[i_X] = ZERO_R1;
}
void kernel fill_ones_int(global symint* a)
{
    a[i_X] = 1;
}
void kernel fill_ones_real(global real1* a)
{
    a[i_X] = ONE_R1;
}
void kernel fill_ones_complex(global cmplx* a)
{
    a[i_X] = (cmplx)(ONE_R1, ZERO_R1);
}
void kernel fill_value_int(global symint* a, constant symint* v)
{
    a[i_X] = *v;
}
void kernel fill_value_real(global real1* a, constant real1* v)
{
    a[i_X] = *v;
}
void kernel fill_value_complex(global cmplx* a, constant cmplx* v)
{
    a[i_X] = *v;
}
void kernel real_to_complex_buffer(global const real1* a, global cmplx* b)
{
    b[i_X] = (cmplx)(a[i_X], ZERO_R1);
}

void kernel reduce_real(global const real1* a, global real1* b, constant tcapint* shape, constant tcapint* stride, constant tcapint* vecCapIntArgs)
{
  tcapint tmp = i_X;
  tcapint base = O_A;
  const tcapint id = I_A;

  for (int d = O_C - 1; d >= 0; --d) {
    if (d == id) {
      continue;
    }

    tcapint dim = shape[d];
    tcapint i_d = tmp % dim;
    tmp /= dim;

    base += i_d * stride[d];
  }

  real1 sum = ZERO_R1;
  for (tcapint j = 0U; j < shape[id]; ++j) {
    sum += a[base + j * stride[id]];
  }
  b[i_X * I_B + O_B] = sum;
}
void kernel reduce_complex(global const cmplx* a, global cmplx* b, constant tcapint* shape, constant tcapint* stride, constant tcapint* vecCapIntArgs)
{
  tcapint tmp = i_X;
  tcapint base = O_A;
  const tcapint id = I_A;

  for (int d = O_C - 1; d >= 0; --d) {
    if (d == id) {
      continue;
    }

    tcapint dim = shape[d];
    tcapint i_d = tmp % dim;
    tmp /= dim;

    base += i_d * stride[d];
  }

  cmplx sum = (cmplx)(ZERO_R1);
  for (tcapint j = 0U; j < shape[id]; ++j) {
    sum += a[base + j * stride[id]];
  }
  b[i_X * I_B + O_B] = sum;
}
void kernel reduce_grad_real(global real1* din, global const real1* dout, constant tcapint* shape, constant tcapint* stride, constant tcapint* vecCapIntArgs)
{
  tcapint tmp = i_X;
  tcapint o = 0U;
  const tcapint id = I_A;

  for (int d = O_C - 1; d >= 0; --d) {
    if (d == id) {
      continue;
    }

    tcapint dim = shape[d];
    tcapint i_d = tmp % dim;
    tmp /= dim;

    o += i_d * stride[d];
  }

  din[i_X * I_B + O_B] += dout[o];
}
void kernel reduce_grad_complex(global cmplx* din, global const cmplx* dout, constant tcapint* shape, constant tcapint* stride, constant tcapint* vecCapIntArgs)
{
  tcapint tmp = i_X;
  tcapint o = 0U;
  const tcapint id = I_A;

  for (int d = O_C - 1; d >= 0; --d) {
    if (d == id) {
      continue;
    }

    tcapint dim = shape[d];
    tcapint i_d = tmp % dim;
    tmp /= dim;

    o += i_d * stride[d];
  }

  din[i_X * I_B + O_B] += dout[o];
}
void kernel reduce_grad_mixed(global real1* din, global const real1* dout, constant tcapint* shape, constant tcapint* stride, constant tcapint* vecCapIntArgs)
{
  tcapint tmp = i_X;
  tcapint o = 0U;
  const tcapint id = I_A;

  for (int d = O_C - 1; d >= 0; --d) {
    if (d == id) {
      continue;
    }

    tcapint dim = shape[d];
    tcapint i_d = tmp % dim;
    tmp /= dim;

    o += i_d * stride[d];
  }

  din[(i_X * I_B + O_B) << 1U] += dout[o];
}
void kernel axis_max(global const real1* a, global real1* b, constant tcapint* shape, constant tcapint* stride, constant tcapint* vecCapIntArgs)
{
  tcapint tmp = i_X;
  tcapint base = O_A;
  const tcapint id = I_A;

  for (int d = O_C - 1; d >= 0; --d) {
    if (d == id) {
      continue;
    }

    tcapint dim = shape[d];
    tcapint i_d = tmp % dim;
    tmp /= dim;

    base += i_d * stride[d];
  }

  real1 m = a[base];
  for (tcapint j = 1U; j < shape[id]; ++j) {
    const real1 v = a[base + j * stride[id]];
    if (v > m) {
      m = v;
    }
  }
  b[i_X * I_B + O_B] = m;
}
void kernel axis_min(global const real1* a, global real1* b, constant tcapint* shape, constant tcapint* stride, constant tcapint* vecCapIntArgs)
{
  tcapint tmp = i_X;
  tcapint base = O_A;
  const tcapint id = I_A;

  for (int d = O_C - 1; d >= 0; --d) {
    if (d == id) {
      continue;
    }

    tcapint dim = shape[d];
    tcapint i_d = tmp % dim;
    tmp /= dim;

    base += i_d * stride[d];
  }

  real1 m = a[base];
  for (tcapint j = 1U; j < shape[id]; ++j) {
    const real1 v = a[base + j * stride[id]];
    if (v < m) {
      m = v;
    }
  }
  b[i_X * I_B + O_B] = m;
}
void kernel axis_match_grad_real(global real1* din, global const real1* dout, global const real1* in, global const real1* out, constant tcapint* shape, constant tcapint* stride, constant tcapint* vecCapIntArgs)
{
  tcapint tmp = i_X;
  tcapint o = 0U;
  const tcapint id = I_A;

  for (int d = O_C - 1; d >= 0; --d) {
    if (d == id) {
      continue;
    }

    tcapint dim = shape[d];
    tcapint i_d = tmp % dim;
    tmp /= dim;

    o += i_d * stride[d];
  }

  const size_t i = i_X * I_B + O_B;
  if (in[i] == out[o]) {
    din[i] += dout[o];
  }
}
void kernel axis_match_grad_complex(global cmplx* din, global const cmplx* dout, global const real1* in, global const real1* out, constant tcapint* shape, constant tcapint* stride, constant tcapint* vecCapIntArgs)
{
  tcapint tmp = i_X;
  tcapint o = 0U;
  const tcapint id = I_A;

  for (int d = O_C - 1; d >= 0; --d) {
    if (d == id) {
      continue;
    }

    tcapint dim = shape[d];
    tcapint i_d = tmp % dim;
    tmp /= dim;

    o += i_d * stride[d];
  }

  const size_t i = i_X * I_B + O_B;
  if (in[i] == out[o]) {
    din[i] += dout[o];
  }
}
void kernel axis_match_grad_mixed(global real1* din, global const real1* dout, global const real1* in, global const real1* out, constant tcapint* shape, constant tcapint* stride, constant tcapint* vecCapIntArgs)
{
  tcapint tmp = i_X;
  tcapint o = 0U;
  const tcapint id = I_A;

  for (int d = O_C - 1; d >= 0; --d) {
    if (d == id) {
      continue;
    }

    tcapint dim = shape[d];
    tcapint i_d = tmp % dim;
    tmp /= dim;

    o += i_d * stride[d];
  }

  const size_t i = i_X * I_B + O_B;
  if (in[i] == out[o]) {
    din[i << 1U] += dout[o];
  }
}

void kernel relu(global const real1* a, global real1* out, constant tcapint* vecCapIntArgs)
{
    const real1 tmp = a[i_X * I_A + O_A];
    out[i_X * I_B + O_B] = (tmp > 0) ? tmp : ZERO_R1;
}
void kernel relu_grad_real(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs)
{
    if (in[i_X * I_B + O_B] > ZERO_R1) {
      din[i_X * I_A + O_A] += dout[i_X * I_C + O_C];
    }
}
void kernel relu_grad_complex(global cmplx* din, global const real1* in, global const cmplx* dout, constant tcapint* vecCapIntArgs)
{
    if (in[i_X * I_B + O_B] > ZERO_R1) {
      din[i_X * I_A + O_A] += dout[i_X * I_C + O_C];
    }
}
void kernel relu_grad_mixed(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs)
{
    if (in[i_X * I_B + O_B] > 0) {
      din[(i_X * I_A + O_A) << 1U] += dout[i_X * I_C + O_C];
    }
}

void kernel sigmoid(global const real1* a, global real1* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_B + O_B] = ONE_R1 / (ONE_R1 + (real1)exp((real1_f)(-a[i_X * I_A + O_A])));
}
void kernel sigmoid_grad_real(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs)
{
    const real1 yi = in[i_X * I_B + O_B];
    din[i_X * I_A + O_A] += yi * (ONE_R1 - yi) * dout[i_X * I_C + O_C];
}
void kernel sigmoid_grad_complex(global cmplx* din, global const real1* in, global const cmplx* dout, constant tcapint* vecCapIntArgs)
{
    const real1 yi = in[i_X * I_B + O_B];
    din[i_X * I_A + O_A] += yi * (ONE_R1 - yi) * dout[i_X * I_C + O_C];
}
void kernel sigmoid_grad_mixed(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs)
{
    const real1 yi = in[i_X * I_B + O_B];
    din[(i_X * I_A + O_A) << 1U] += yi * (ONE_R1 - yi) * dout[i_X * I_C + O_C];
}

void kernel wtanh(global const real1* a, global real1* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_B + O_B] = tanh(a[i_X * I_A + O_A]);
}
void kernel wtanh_grad_real(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs)
{
    const real1 yi = in[i_X * I_B + O_B];
    din[i_X * I_A + O_A] += (ONE_R1 - yi * yi) * dout[i_X * I_C + O_C];
}
void kernel wtanh_grad_complex(global cmplx* din, global const real1* in, global const cmplx* dout, constant tcapint* vecCapIntArgs)
{
    const real1 yi = in[i_X * I_B + O_B];
    din[i_X * I_A + O_A] += (ONE_R1 - yi * yi) * dout[i_X * I_C + O_C];
}
void kernel wtanh_grad_mixed(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs)
{
    const real1 yi = in[i_X * I_B + O_B];
    din[(i_X * I_A + O_A) << 1U] += (ONE_R1 - yi * yi) * dout[i_X * I_C + O_C];
}

void kernel wsin(global const real1* a, global real1* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_B + O_B] = sin(a[i_X * I_A + O_A]);
}
void kernel wsin_grad_real(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs)
{
    din[i_X * I_A + O_A] += cos(in[i_X * I_B + O_B]) * dout[i_X * I_C + O_C];
}
void kernel wsin_grad_complex(global cmplx* din, global const real1* in, global const cmplx* dout, constant tcapint* vecCapIntArgs)
{
    din[i_X * I_A + O_A] += cos(in[i_X * I_B + O_B]) * dout[i_X * I_C + O_C];
}
void kernel wsin_grad_mixed(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs)
{
    din[i_X * I_A + O_A] += cos(in[i_X * I_B + O_B]) * dout[i_X * I_C + O_C];
}

void kernel wcos(global const real1* a, global real1* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_B + O_B] = cos(a[i_X * I_A + O_A]);
}
void kernel wcos_grad_real(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs)
{
    din[i_X * I_A + O_A] += -sin(in[i_X * I_B + O_B]) * dout[i_X * I_C + O_C];
}
void kernel wcos_grad_complex(global cmplx* din, global const real1* in, global const cmplx* dout, constant tcapint* vecCapIntArgs)
{
    din[i_X * I_A + O_A] += -sin(in[i_X * I_B + O_B]) * dout[i_X * I_C + O_C];
}
void kernel wcos_grad_mixed(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs)
{
    din[i_X * I_A + O_A] += -sin(in[i_X * I_B + O_B]) * dout[i_X * I_C + O_C];
}

void kernel match_grad_real(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs, constant real1* m)
{
    if (*m == in[i_X * I_B + O_B]) {
      din[i_X * I_A + O_A] += dout[i_X * I_C + O_C];
    }
}
void kernel match_grad_complex(global cmplx* din, global const real1* in, global const cmplx* dout, constant tcapint* vecCapIntArgs, constant real1* m)
{
    if (*m == in[i_X * I_B + O_B]) {
      din[i_X * I_A + O_A] += dout[i_X * I_C + O_C];
    }
}
void kernel match_grad_mixed(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs, constant real1* m)
{
    if (*m == in[i_X * I_B + O_B]) {
      din[(i_X * I_A + O_A) << 1U] += dout[i_X * I_C + O_C];
    }
}

void kernel clamp_real(global const real1* a, global real1* out, constant tcapint* vecCapIntArgs, constant cmplx* p)
{
    const real1 tmp = a[i_X * I_A + O_A];
    const cmplx b = *p;
    out[i_X * I_B + O_B] = (tmp > b.x) ? ((tmp < b.y) ? tmp  : b.y) : b.x;
}
void kernel clamp_grad_real(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs, constant cmplx* p)
{
    const real1 xi = in[i_X * I_B + O_B];
    const cmplx b = *p;
    if (xi > b.x && xi < b.y) {
      din[i_X * I_C + O_C] += dout[i_X * I_A + O_A];
    }
}
void kernel clamp_grad_complex(global cmplx* din, global const real1* in, global const cmplx* dout, constant tcapint* vecCapIntArgs, constant cmplx* p)
{
    const real1 xi = in[i_X * I_B + O_B];
    const cmplx b = *p;
    if (xi > b.x && xi < b.y) {
      din[i_X * I_C + O_C] += dout[i_X * I_A + O_A];
    }
}
void kernel clamp_grad_mixed(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs, constant cmplx* p)
{
    const real1 xi = in[i_X * I_B + O_B];
    const cmplx b = *p;
    if (xi > b.x && xi < b.y) {
      din[(i_X * I_C + O_C) << 1U] += dout[i_X * I_A + O_A];
    }
}

void kernel abs_real(global const real1* a, global real1* out, constant tcapint* vecCapIntArgs)
{
    const real1 tmp = a[i_X * I_A + O_A];
    out[i_X * I_B + O_B] = (tmp < 0) ? -tmp : tmp;
}
void kernel abs_complex(global const cmplx* a, global real1* out, constant tcapint* vecCapIntArgs)
{
    const cmplx tmp = a[i_X * I_A + O_A];
    out[i_X * I_B + O_B] = sqrt(dot(tmp, tmp));
}
void kernel abs_real_grad_real(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs)
{
    const real1 tmp = in[i_X * I_B + O_B];
    if (tmp != ZERO_R1) {
      const real1 tmp_o = dout[i_X * I_C + O_C];
      din[i_X * I_A + O_A] += (tmp > ZERO_R1) ? tmp_o : -tmp_o;
    }
}
void kernel abs_real_grad_complex(global cmplx* din, global const real1* in, global const cmplx* dout, constant tcapint* vecCapIntArgs)
{
    const real1 tmp = in[i_X * I_B + O_B];
    if (tmp != ZERO_R1) {
      const cmplx tmp_o = dout[i_X * I_C + O_C];
      din[i_X * I_A + O_A] += (tmp > ZERO_R1) ? tmp_o : -tmp_o;
    }
}
void kernel abs_real_grad_mixed(global real1* din, global const real1* in, global const real1* dout, constant tcapint* vecCapIntArgs)
{
    const real1 tmp = in[i_X * I_B + O_B];
    if (tmp != ZERO_R1) {
      const real1 tmp_o = dout[i_X * I_C + O_C];
      din[(i_X * I_A + O_A) << 1U] += (tmp > ZERO_R1) ? tmp_o : -tmp_o;
    }
}
void kernel abs_complex_grad_real(global cmplx* din, global const cmplx* in, global const real1* dout, constant tcapint* vecCapIntArgs)
{
    const cmplx tmp = in[i_X * I_B + O_B];
    if ((tmp.x != ZERO_R1) || (tmp.y != ZERO_R1)) {
      din[i_X * I_A + O_A] += (dout[i_X * I_C + O_C] / sqrt(dot(tmp, tmp))) * tmp;
    }
}
void kernel abs_complex_grad_complex(global cmplx* din, global const cmplx* in, global const cmplx* dout, constant tcapint* vecCapIntArgs)
{
    const cmplx tmp = in[i_X * I_B + O_B];
    if ((tmp.x != ZERO_R1) || (tmp.y != ZERO_R1)) {
      din[i_X * I_A + O_A] += zmul(dout[i_X * I_C + O_C], tmp) / sqrt(dot(tmp, tmp));
    }
}
void kernel abs_complex_grad_mixed(global cmplx* din, global const cmplx* in, global const real1* dout, constant tcapint* vecCapIntArgs)
{
    const cmplx tmp = in[i_X * I_B + O_B];
    if ((tmp.x != ZERO_R1) || (tmp.y != ZERO_R1)) {
      din[i_X * I_A + O_A] += (dout[i_X * I_C + O_C] / sqrt(dot(tmp, tmp))) * tmp;
    }
}

void kernel add_real(global const real1* a, global const real1* b, global real1* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = a[i_X * I_A + O_A] + b[i_X * I_B + O_B];
}
void kernel add_complex(global const cmplx* a, global const cmplx* b, global cmplx* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = a[i_X * I_A + O_A] + b[i_X * I_B + O_B];
}
void kernel add_mixed(global const cmplx* a, global const real1* b, global cmplx* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = a[i_X * I_A + O_A] + (cmplx)(b[i_X * I_B + O_B], 0);
}

void kernel mul_real(global const real1* a, global const real1* b, global real1* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = a[i_X * I_A + O_A] * b[i_X * I_B + O_B];
}
void kernel mul_complex(global const cmplx* a, global const cmplx* b, global cmplx* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = zmul(a[i_X * I_A + O_A], b[i_X * I_B + O_B]);
}
void kernel mul_mixed(global const cmplx* a, global const real1* b, global cmplx* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = b[i_X * I_B + O_B] * a[i_X * I_A + O_A];
}

void kernel matmul_real(
    global const real1* a,
    global const real1* b,
    global real1* out,
    constant tcapint* vecCapIntArgs)
{
    const tcapint tile_row = get_local_id(0);
    const tcapint tile_col = get_local_id(1);
    const tcapint global_row = get_group_id(0) * TILE_SIZE + tile_row;
    const tcapint global_col = get_group_id(1) * TILE_SIZE + tile_col;

    local real1 tile_a[TILE_SIZE][TILE_SIZE];
    local real1 tile_b[TILE_SIZE][TILE_SIZE];

    real1 sum = ZERO_R1;

    const tcapint num_tiles = (K + TILE_SIZE - 1U) / TILE_SIZE;

    for (tcapint t = 0U; t < num_tiles; ++t) {
        const tcapint a_col = t * TILE_SIZE + tile_col;
        tile_a[tile_row][tile_col] =
            (global_row < M && a_col < K)
            ? a[O_A + global_row * I_A + a_col * J_A]
            : ZERO_R1;

        const tcapint b_row = t * TILE_SIZE + tile_row;
        tile_b[tile_row][tile_col] =
            (b_row < K && global_col < N)
            ? b[O_B + b_row * I_B + global_col * J_B]
            : ZERO_R1;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (tcapint k = 0U; k < TILE_SIZE; ++k) {
            sum = fma(tile_a[tile_row][k], tile_b[k][tile_col], sum);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N) {
        out[O_C + global_row * I_C + global_col * J_C] = sum;
    }
}
#define TILE_SIZE 16

void kernel matmul_complex(
    global const real1* a,
    global const real1* b,
    global real1* out,
    constant tcapint* vecCapIntArgs)
{
    const tcapint tile_row = get_local_id(0);
    const tcapint tile_col = get_local_id(1);
    const tcapint global_row = get_group_id(0) * TILE_SIZE + tile_row;
    const tcapint global_col = get_group_id(1) * TILE_SIZE + tile_col;

    // Local tiles store interleaved real/imag pairs
    // tile_a[row][col] = {real, imag} at that position
    local real1 tile_a_re[TILE_SIZE][TILE_SIZE];
    local real1 tile_a_im[TILE_SIZE][TILE_SIZE];
    local real1 tile_b_re[TILE_SIZE][TILE_SIZE];
    local real1 tile_b_im[TILE_SIZE][TILE_SIZE];

    real1 sum_re = ZERO_R1;
    real1 sum_im = ZERO_R1;

    const tcapint num_tiles = (K + TILE_SIZE - 1U) / TILE_SIZE;

    for (tcapint t = 0U; t < num_tiles; ++t) {
        // Load tile of A — complex elements are stride-2 in storage
        const tcapint a_col = t * TILE_SIZE + tile_col;
        if (global_row < M && a_col < K) {
            const tcapint a_idx = (O_A + global_row * I_A + a_col * J_A) << 1U;
            tile_a_re[tile_row][tile_col] = a[a_idx];
            tile_a_im[tile_row][tile_col] = a[a_idx + 1U];
        } else {
            tile_a_re[tile_row][tile_col] = ZERO_R1;
            tile_a_im[tile_row][tile_col] = ZERO_R1;
        }

        // Load tile of B
        const tcapint b_row = t * TILE_SIZE + tile_row;
        if (b_row < K && global_col < N) {
            const tcapint b_idx = (O_B + b_row * I_B + global_col * J_B) << 1U;
            tile_b_re[tile_row][tile_col] = b[b_idx];
            tile_b_im[tile_row][tile_col] = b[b_idx + 1U];
        } else {
            tile_b_re[tile_row][tile_col] = ZERO_R1;
            tile_b_im[tile_row][tile_col] = ZERO_R1;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        for (tcapint k = 0U; k < TILE_SIZE; ++k) {
            const real1 ar = tile_a_re[tile_row][k];
            const real1 ai = tile_a_im[tile_row][k];
            const real1 br = tile_b_re[k][tile_col];
            const real1 bi = tile_b_im[k][tile_col];
            sum_re = fma(ar, br, fma(-ai, bi, sum_re));  // ac - bd
            sum_im = fma(ar, bi, fma( ai, br, sum_im));  // ad + bc
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N) {
        const tcapint o_idx = (O_C + global_row * I_C + global_col * J_C) << 1U;
        out[o_idx]      = sum_re;
        out[o_idx + 1U] = sum_im;
    }
}
void kernel matmul_mixed_c_left(
    global const real1* a,
    global const real1* b,
    global real1* out,
    constant tcapint* vecCapIntArgs)
{
    const tcapint tile_row = get_local_id(0);
    const tcapint tile_col = get_local_id(1);
    const tcapint global_row = get_group_id(0) * TILE_SIZE + tile_row;
    const tcapint global_col = get_group_id(1) * TILE_SIZE + tile_col;

    // Local tiles store interleaved real/imag pairs
    // tile_a[row][col] = {real, imag} at that position
    local real1 tile_a_re[TILE_SIZE][TILE_SIZE];
    local real1 tile_a_im[TILE_SIZE][TILE_SIZE];
    local real1 tile_b_re[TILE_SIZE][TILE_SIZE];
    local real1 tile_b_im[TILE_SIZE][TILE_SIZE];

    real1 sum_re = ZERO_R1;
    real1 sum_im = ZERO_R1;

    const tcapint num_tiles = (K + TILE_SIZE - 1U) / TILE_SIZE;

    for (tcapint t = 0U; t < num_tiles; ++t) {
        // Load tile of A — complex elements are stride-2 in storage
        const tcapint a_col = t * TILE_SIZE + tile_col;
        if (global_row < M && a_col < K) {
            const tcapint a_idx = (O_A + global_row * I_A + a_col * J_A) << 1U;
            tile_a_re[tile_row][tile_col] = a[a_idx];
            tile_a_im[tile_row][tile_col] = a[a_idx + 1U];
        } else {
            tile_a_re[tile_row][tile_col] = ZERO_R1;
            tile_a_im[tile_row][tile_col] = ZERO_R1;
        }

        // Load tile of B
        const tcapint b_row = t * TILE_SIZE + tile_row;
        if (b_row < K && global_col < N) {
            const tcapint b_idx = (O_B + b_row * I_B + global_col * J_B) << 1U;
            tile_b_re[tile_row][tile_col] = b[b_idx];
            tile_b_im[tile_row][tile_col] = b[b_idx + 1U];
        } else {
            tile_b_re[tile_row][tile_col] = ZERO_R1;
            tile_b_im[tile_row][tile_col] = ZERO_R1;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        for (tcapint k = 0U; k < TILE_SIZE; ++k) {
            const real1 ar = tile_a_re[tile_row][k];
            const real1 ai = tile_a_im[tile_row][k];
            const real1 br = tile_b_re[k][tile_col];
            const real1 bi = tile_b_im[k][tile_col];
            // bi = 0, so:
            sum_re = fma(ar, br, sum_re);   // ac
            sum_im = fma(ai, br, sum_im);   // bc
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N) {
        const tcapint o_idx = (O_C + global_row * I_C + global_col * J_C) << 1U;
        out[o_idx]      = sum_re;
        out[o_idx + 1U] = sum_im;
    }
}
void kernel matmul_mixed_c_right(
    global const real1* a,
    global const real1* b,
    global real1* out,
    constant tcapint* vecCapIntArgs)
{
    const tcapint tile_row = get_local_id(0);
    const tcapint tile_col = get_local_id(1);
    const tcapint global_row = get_group_id(0) * TILE_SIZE + tile_row;
    const tcapint global_col = get_group_id(1) * TILE_SIZE + tile_col;

    // Local tiles store interleaved real/imag pairs
    // tile_a[row][col] = {real, imag} at that position
    local real1 tile_a_re[TILE_SIZE][TILE_SIZE];
    local real1 tile_a_im[TILE_SIZE][TILE_SIZE];
    local real1 tile_b_re[TILE_SIZE][TILE_SIZE];
    local real1 tile_b_im[TILE_SIZE][TILE_SIZE];

    real1 sum_re = ZERO_R1;
    real1 sum_im = ZERO_R1;

    const tcapint num_tiles = (K + TILE_SIZE - 1U) / TILE_SIZE;

    for (tcapint t = 0U; t < num_tiles; ++t) {
        // Load tile of A — complex elements are stride-2 in storage
        const tcapint a_col = t * TILE_SIZE + tile_col;
        if (global_row < M && a_col < K) {
            const tcapint a_idx = (O_A + global_row * I_A + a_col * J_A) << 1U;
            tile_a_re[tile_row][tile_col] = a[a_idx];
            tile_a_im[tile_row][tile_col] = a[a_idx + 1U];
        } else {
            tile_a_re[tile_row][tile_col] = ZERO_R1;
            tile_a_im[tile_row][tile_col] = ZERO_R1;
        }

        // Load tile of B
        const tcapint b_row = t * TILE_SIZE + tile_row;
        if (b_row < K && global_col < N) {
            const tcapint b_idx = (O_B + b_row * I_B + global_col * J_B) << 1U;
            tile_b_re[tile_row][tile_col] = b[b_idx];
            tile_b_im[tile_row][tile_col] = b[b_idx + 1U];
        } else {
            tile_b_re[tile_row][tile_col] = ZERO_R1;
            tile_b_im[tile_row][tile_col] = ZERO_R1;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        for (tcapint k = 0U; k < TILE_SIZE; ++k) {
            const real1 ar = tile_a_re[tile_row][k];
            const real1 ai = tile_a_im[tile_row][k];
            const real1 br = tile_b_re[k][tile_col];
            const real1 bi = tile_b_im[k][tile_col];
            // ai = 0, so:
            sum_re = fma(ar, br, sum_re);   // ac
            sum_im = fma(ar, bi, sum_im);   // ad
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_row < M && global_col < N) {
        const tcapint o_idx = (O_C + global_row * I_C + global_col * J_C) << 1U;
        out[o_idx]      = sum_re;
        out[o_idx + 1U] = sum_im;
    }
}

void kernel sub_real(global const real1* a, global const real1* b, global real1* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = a[i_X * I_A + O_A] - b[i_X * I_B + O_B];
}
void kernel sub_complex(global const cmplx* a, global const cmplx* b, global cmplx* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = a[i_X * I_A + O_A] - b[i_X * I_B + O_B];
}
void kernel sub_mixed_c_left(global const cmplx* a, global const real1* b, global cmplx* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = a[i_X * I_A + O_A] - (cmplx)(b[i_X * I_B + O_B], ZERO_R1);
}
void kernel sub_mixed_c_right(global const real1* a, global const cmplx* b, global cmplx* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = (cmplx)(a[i_X * I_A + O_A], ZERO_R1) - b[i_X * I_B + O_B];
}

void kernel div_real(global const real1* a, global const real1* b, global real1* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = a[i_X * I_A + O_A] / b[i_X * I_B + O_B];
}
void kernel div_complex(global const cmplx* a, global const cmplx* b, global cmplx* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = zdiv(a[i_X * I_A + O_A], b[i_X * I_B + O_B]);
}
void kernel div_mixed_c_left(global const cmplx* a, global const real1* b, global cmplx* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = a[i_X * I_A + O_A] / b[i_X * I_B + O_B];
}
void kernel div_mixed_c_right(global const real1* a, global const cmplx* b, global cmplx* out, constant tcapint* vecCapIntArgs)
{
    out[i_X * I_C + O_C] = zdiv((cmplx)(a[i_X * I_A + O_A], ZERO_R1), b[i_X * I_B + O_B]);
}

void kernel add_in_place_real(global real1* a, global const real1* b, constant tcapint* vecCapIntArgs)
{
    a[i_X * I_A + O_A] += b[i_X * I_B + O_B];
}
void kernel add_in_place_complex(global cmplx* a, global const cmplx* b, constant tcapint* vecCapIntArgs)
{
    a[i_X * I_A + O_A] += b[i_X * I_B + O_B];
}
void kernel add_in_place_mixed(global cmplx* a, global const real1* b, constant tcapint* vecCapIntArgs)
{
    a[i_X * I_A + O_A] += (cmplx)(b[i_X * I_B + O_B], 0);
}

void kernel sub_in_place_real(global real1* a, global const real1* b, constant tcapint* vecCapIntArgs)
{
    a[i_X * I_A + O_A] -= b[i_X * I_B + O_B];
}
void kernel sub_in_place_complex(global cmplx* a, global const cmplx* b, constant tcapint* vecCapIntArgs)
{
    a[i_X * I_A + O_A] -= b[i_X * I_B + O_B];
}
void kernel sub_in_place_mixed(global cmplx* a, global const real1* b, constant tcapint* vecCapIntArgs)
{
    a[i_X * I_A + O_A] -= (cmplx)(b[i_X * I_B + O_B], 0);
}

void kernel pow_real(global const real1* a, global real1* out, constant tcapint* vecCapIntArgs, constant real1* p)
{
    out[i_X * I_B + O_B] = (real1)pow((real1_f)a[i_X * I_A + O_A], (real1_f)*p);
}
void kernel pow_complex(global const cmplx* a, global cmplx* out, constant tcapint* vecCapIntArgs, constant real1* p)
{
    out[i_X * I_B + O_B] = zpow_real(a[i_X * I_A + O_A], *p);
}
void kernel exp_real(global const real1* a, global real1* out, constant tcapint* vecCapIntArgs, constant real1* log_b)
{
    out[i_X * I_B + O_B] = ((real1)exp((real1_f)(a[i_X * I_A + O_A]) * (*log_b)));
}
void kernel exp_complex(global const cmplx* a, global cmplx* out, constant tcapint* vecCapIntArgs, constant real1* log_b)
{
    out[i_X * I_B + O_B] = zexp(a[i_X * I_A + O_A] * (*log_b));
}
void kernel log_real(global const real1* a, global real1* out, constant tcapint* vecCapIntArgs, constant real1* inv_log_b)
{
    out[i_X * I_B + O_B] = ((real1)log((real1_f)a[i_X * I_A + O_A])) * (*inv_log_b);
}
void kernel log_complex(global const cmplx* a, global cmplx* out, constant tcapint* vecCapIntArgs, constant real1* inv_log_b)
{
    out[i_X * I_B + O_B] = zlog(a[i_X * I_A + O_A]) * (*inv_log_b);
}

void kernel embedding_real(global const symint* idx, global const real1* W, global real1* O, constant tcapint* vecCapIntArgs)
{
    const tcapint token = idx[O_A + i_X * I_A];
    const tcapint w_base = O_B + token * J_C;
    const tcapint o_base = J_A + i_X * I_B;
    for (tcapint d = 0U; d < J_B; ++d) {
      O[o_base + d * I_C] = W[w_base + d * O_C];
    }
}
void kernel embedding_complex(global const symint* idx, global const cmplx* W, global cmplx* O, constant tcapint* vecCapIntArgs)
{
    const tcapint token = idx[O_A + i_X * I_A];
    const tcapint w_base = O_B + token * J_C;
    const tcapint o_base = J_A + i_X * I_B;
    for (tcapint d = 0U; d < J_B; ++d) {
      O[o_base + d * I_C] = W[w_base + d * O_C];
    }
}
void kernel embedding_grad_real(global const symint* idx, global real1* dW, global const real1* dO, constant tcapint* vecCapIntArgs)
{
    const tcapint token = idx[O_A + i_X * I_A];
    const tcapint w_base = O_B + token * J_C;
    const tcapint o_base = J_A + i_X * I_B;
    for (tcapint d = 0U; d < J_B; ++d) {
      dW[w_base + d * O_C] += dO[o_base + d * I_C];
    }
}
void kernel embedding_grad_complex(global const symint* idx, global cmplx* dW, global const cmplx* dO, constant tcapint* vecCapIntArgs)
{
    const tcapint token = idx[O_A + i_X * I_A];
    const tcapint w_base = O_B + token * J_C;
    const tcapint o_base = J_A + i_X * I_B;
    for (tcapint d = 0U; d < J_B; ++d) {
      dW[w_base + d * O_C] += dO[o_base + d * I_C];
    }
}
void kernel embedding_grad_mixed(global const symint* idx, global real1* dW, global const real1* dO, constant tcapint* vecCapIntArgs)
{
    const tcapint token = idx[O_A + i_X * I_A];
    const tcapint w_base = O_B + token * J_C;
    const tcapint o_base = J_A + i_X * I_B;
    for (tcapint d = 0U; d < J_B; ++d) {
      dW[(w_base + d * O_C) << 1U] += dO[o_base + d * I_C];
    }
}

void kernel copy_real(global real1* a, global const real1* b, constant tcapint* vecCapIntArgs)
{
    a[i_X * I_A + O_A] = b[i_X * I_B + O_B];
}
void kernel copy_complex(global cmplx* a, global const cmplx* b, constant tcapint* vecCapIntArgs)
{
    a[i_X * I_A + O_A] = b[i_X * I_B + O_B];
}

void kernel triu_fill_real(global real1* a, global const cmplx* val, constant tcapint* vecCapIntArgs)
{
    if ((i_X + O_A) <= i_Y) {
      a[i_X * S_X + i_Y] = val->x;
    }
}
void kernel triu_fill_complex(global cmplx* a, global const cmplx* val, constant tcapint* vecCapIntArgs)
{
    if ((i_X + O_A) <= i_Y) {
      a[i_X * S_X + i_Y] = *val;
    }
}

kernel void softmax_forward(global const real1 *x, global real1 *out, global const tcapint *vciArgs)
{
    const tcapint x_off      = vciArgs[0];
    const tcapint x_stride   = vciArgs[1];
    const tcapint axis_size  = vciArgs[2];
    const tcapint out_off    = vciArgs[3];
    const tcapint out_stride = vciArgs[4];
    const tcapint outer_str  = vciArgs[5];

    const tcapint outer = get_global_id(0);

    const tcapint x_base   = x_off   + outer * outer_str;
    const tcapint out_base = out_off  + outer * outer_str;

    // Pass 1: find max
    real1 mx = x[x_base];
    for (tcapint i = 1; i < axis_size; ++i) {
        const real1 v = x[x_base + i * x_stride];
        if (v > mx) mx = v;
    }

    // Pass 2: sum of exp(x - max)
    real1 s = ZERO_R1;
    for (tcapint i = 0; i < axis_size; ++i) {
        s += exp(x[x_base + i * x_stride] - mx);
    }

    // Pass 3: write normalized output
    for (tcapint i = 0; i < axis_size; ++i) {
        out[out_base + i * out_stride] =
            exp(x[x_base + i * x_stride] - mx) / s;
    }
}
kernel void softmax_backward_real(global real1 *dx, global const real1 *out, global const real1 *dy, global const tcapint *vciArgs)
{
    const tcapint dx_off     = vciArgs[0];
    const tcapint dx_stride  = vciArgs[1];
    const tcapint axis_size  = vciArgs[2];
    const tcapint out_off    = vciArgs[3];
    const tcapint out_stride = vciArgs[4];
    const tcapint dy_off     = vciArgs[5];
    const tcapint dy_stride  = vciArgs[6];
    const tcapint outer_str  = vciArgs[7];

    const tcapint outer = get_global_id(0);

    const tcapint dx_base  = dx_off  + outer * outer_str;
    const tcapint out_base = out_off + outer * outer_str;
    const tcapint dy_base  = dy_off  + outer * outer_str;

    // dot = sum(dy * out)
    real1 dt = ZERO_R1;
    for (tcapint i = 0; i < axis_size; ++i) {
        dt += dy[dy_base + i * dy_stride] * out[out_base + i * out_stride];
    }

    // dx += out * (dy - dot)
    for (tcapint i = 0; i < axis_size; ++i) {
        dx[dx_base + i * dx_stride] +=
            out[out_base + i * out_stride] *
            (dy[dy_base + i * dy_stride] - dt);
    }
}
kernel void softmax_backward_complex(global cmplx *dx, global const real1 *out, global const cmplx *dy, global const tcapint *vciArgs)
{
    const tcapint dx_off     = vciArgs[0];
    const tcapint dx_stride  = vciArgs[1];
    const tcapint axis_size  = vciArgs[2];
    const tcapint out_off    = vciArgs[3];
    const tcapint out_stride = vciArgs[4];
    const tcapint dy_off     = vciArgs[5];
    const tcapint dy_stride  = vciArgs[6];
    const tcapint outer_str  = vciArgs[7];

    const tcapint outer = get_global_id(0);

    const tcapint dx_base  = dx_off  + outer * outer_str;
    const tcapint out_base = out_off + outer * outer_str;
    const tcapint dy_base  = dy_off  + outer * outer_str;

    // dot = sum(dy * out)
    cmplx dt = (cmplx)(ZERO_R1);
    for (tcapint i = 0; i < axis_size; ++i) {
        dt += dy[dy_base + i * dy_stride] * out[out_base + i * out_stride];
    }

    // dx += out * (dy - dot)
    for (tcapint i = 0; i < axis_size; ++i) {
        dx[dx_base + i * dx_stride] +=
            out[out_base + i * out_stride] *
            (dy[dy_base + i * dy_stride] - dt);
    }
}
kernel void softmax_backward_mixed(global cmplx *dx, global const real1 *out, global const real1 *dy, global const tcapint *vciArgs)
{
    const tcapint dx_off     = vciArgs[0];
    const tcapint dx_stride  = vciArgs[1];
    const tcapint axis_size  = vciArgs[2];
    const tcapint out_off    = vciArgs[3];
    const tcapint out_stride = vciArgs[4];
    const tcapint dy_off     = vciArgs[5];
    const tcapint dy_stride  = vciArgs[6];
    const tcapint outer_str  = vciArgs[7];

    const tcapint outer = get_global_id(0);

    const tcapint dx_base  = dx_off  + outer * outer_str;
    const tcapint out_base = out_off + outer * outer_str;
    const tcapint dy_base  = dy_off  + outer * outer_str;

    // dot = sum(dy * out)
    real1 dt = ZERO_R1;
    for (tcapint i = 0; i < axis_size; ++i) {
        dt += dy[dy_base + i * dy_stride] * out[out_base + i * out_stride];
    }

    // dx += out * (dy - dot)
    for (tcapint i = 0; i < axis_size; ++i) {
        dx[dx_base + i * dx_stride].x +=
            out[out_base + i * out_stride] *
            (dy[dy_base + i * dy_stride] - dt);
    }
}

kernel void logsoftmax(global const real1 *x, global real1 *out, global const tcapint *vciArgs)
{
    const tcapint x_off      = vciArgs[0];
    const tcapint x_stride   = vciArgs[1];
    const tcapint axis_size  = vciArgs[2];
    const tcapint out_off    = vciArgs[3];
    const tcapint out_stride = vciArgs[4];
    const tcapint outer_str  = vciArgs[5];

    const tcapint outer = get_global_id(0);

    const tcapint x_base   = x_off  + outer * outer_str;
    const tcapint out_base = out_off + outer * outer_str;

    // Pass 1: find max
    real1 mx = x[x_base];
    for (tcapint i = 1; i < axis_size; ++i) {
        const real1 v = x[x_base + i * x_stride];
        if (v > mx) mx = v;
    }

    // Pass 2: sum of exp(x - max), then log
    real1 s = ZERO_R1;
    for (tcapint i = 0; i < axis_size; ++i) {
        s += exp(x[x_base + i * x_stride] - mx);
    }
    const real1 log_s = log(s);

    // Pass 3: write (x - max) - log(sum)
    for (tcapint i = 0; i < axis_size; ++i) {
        out[out_base + i * out_stride] =
            (x[x_base + i * x_stride] - mx) - log_s;
    }
}

kernel void logsoftmax_backward_real(global real1 *dx, global const real1 *out, global const real1 *dy, global const tcapint *vciArgs)
{
    const tcapint dx_off     = vciArgs[0];
    const tcapint dx_stride  = vciArgs[1];
    const tcapint axis_size  = vciArgs[2];
    const tcapint out_off    = vciArgs[3];
    const tcapint out_stride = vciArgs[4];
    const tcapint dy_off     = vciArgs[5];
    const tcapint dy_stride  = vciArgs[6];
    const tcapint outer_str  = vciArgs[7];

    const tcapint outer = get_global_id(0);

    const tcapint dx_base  = dx_off  + outer * outer_str;
    const tcapint out_base = out_off + outer * outer_str;
    const tcapint dy_base  = dy_off  + outer * outer_str;

    // Pass 1: sum(dy) along axis
    real1 sum_dy = ZERO_R1;
    for (tcapint i = 0; i < axis_size; ++i) {
        sum_dy += dy[dy_base + i * dy_stride];
    }

    // Pass 2: dx += dy - exp(lsm) * sum_dy
    for (tcapint i = 0; i < axis_size; ++i) {
        dx[dx_base + i * dx_stride] +=
            dy[dy_base + i * dy_stride] -
            exp(out[out_base + i * out_stride]) * sum_dy;
    }
}

kernel void logsoftmax_backward_complex(global cmplx *dx, global const real1 *out, global const cmplx *dy, global const tcapint *vciArgs)
{
    const tcapint dx_off     = vciArgs[0];
    const tcapint dx_stride  = vciArgs[1];
    const tcapint axis_size  = vciArgs[2];
    const tcapint out_off    = vciArgs[3];
    const tcapint out_stride = vciArgs[4];
    const tcapint dy_off     = vciArgs[5];
    const tcapint dy_stride  = vciArgs[6];
    const tcapint outer_str  = vciArgs[7];

    const tcapint outer = get_global_id(0);

    const tcapint dx_base  = dx_off  + outer * outer_str;
    const tcapint out_base = out_off + outer * outer_str;
    const tcapint dy_base  = dy_off  + outer * outer_str;

    // Pass 1: sum(dy) along axis
    cmplx sum_dy = (cmplx)(ZERO_R1);
    for (tcapint i = 0; i < axis_size; ++i) {
        sum_dy += dy[dy_base + i * dy_stride];
    }

    // Pass 2: dx += dy - exp(lsm) * sum_dy
    for (tcapint i = 0; i < axis_size; ++i) {
        dx[dx_base + i * dx_stride] +=
            dy[dy_base + i * dy_stride] -
            exp(out[out_base + i * out_stride]) * sum_dy;
    }
}

kernel void logsoftmax_backward_mixed(global cmplx *dx, global const real1 *out, global const real1 *dy, global const tcapint *vciArgs)
{
    const tcapint dx_off     = vciArgs[0];
    const tcapint dx_stride  = vciArgs[1];
    const tcapint axis_size  = vciArgs[2];
    const tcapint out_off    = vciArgs[3];
    const tcapint out_stride = vciArgs[4];
    const tcapint dy_off     = vciArgs[5];
    const tcapint dy_stride  = vciArgs[6];
    const tcapint outer_str  = vciArgs[7];

    const tcapint outer = get_global_id(0);

    const tcapint dx_base  = dx_off  + outer * outer_str;
    const tcapint out_base = out_off + outer * outer_str;
    const tcapint dy_base  = dy_off  + outer * outer_str;

    // Pass 1: sum(dy) along axis — real only for mixed
    real1 sum_dy = ZERO_R1;
    for (tcapint i = 0; i < axis_size; ++i) {
        sum_dy += dy[dy_base + i * dy_stride];
    }

    // Pass 2: dx += dy - exp(lsm) * sum_dy
    for (tcapint i = 0; i < axis_size; ++i) {
        dx[dx_base + i * dx_stride].x +=
            dy[dy_base + i * dy_stride] -
            exp(out[out_base + i * out_stride]) * sum_dy;
    }
}
