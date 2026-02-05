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

#include "common/cuda_kernels.cuh"

namespace Weed {

__device__ inline qCudaCmplx zmul(const qCudaCmplx lhs, const qCudaCmplx rhs) {
  return make_qCudaCmplx((lhs.x * rhs.x) - (lhs.y * rhs.y),
                         (lhs.x * rhs.y) + (lhs.y * rhs.x));
}

#if FPPOW > 4
__device__ inline qCudaCmplx2
zmatrixmul(const qCudaReal1 nrm, const qCudaReal1 *lhs, const qCudaCmplx2 rhs) {
  return (make_qCudaCmplx2(nrm * ((lhs[0] * rhs.x) - (lhs[1] * rhs.y) +
                                  (lhs[2] * rhs.z) - (lhs[3] * rhs.w)),
                           nrm * ((lhs[0] * rhs.y) + (lhs[1] * rhs.x) +
                                  (lhs[2] * rhs.w) + (lhs[3] * rhs.z)),
                           nrm * ((lhs[4] * rhs.x) - (lhs[5] * rhs.y) +
                                  (lhs[6] * rhs.z) - (lhs[7] * rhs.w)),
                           nrm * ((lhs[4] * rhs.y) + (lhs[5] * rhs.x) +
                                  (lhs[6] * rhs.w) + (lhs[7] * rhs.z))));
}
#else
__device__ inline void zmatrixmul(const qCudaReal1 nrm, const qCudaReal1 *lhs,
                                  const qCudaCmplx2 rhs, qCudaCmplx2 o) {
  o[0] = (make_qCudaCmplx(nrm * ((lhs[0] * rhs[0].x) - (lhs[1] * rhs[0].y) +
                                 (lhs[2] * rhs[1].x) - (lhs[3] * rhs[1].y)),
                          nrm * ((lhs[0] * rhs[0].y) + (lhs[1] * rhs[0].x) +
                                 (lhs[2] * rhs[1].y) + (lhs[3] * rhs[1].x))));
  o[1] = (make_qCudaCmplx(nrm * ((lhs[4] * rhs[0].x) - (lhs[5] * rhs[0].y) +
                                 (lhs[6] * rhs[1].x) - (lhs[7] * rhs[1].y)),
                          nrm * ((lhs[4] * rhs[0].y) + (lhs[5] * rhs[0].x) +
                                 (lhs[6] * rhs[1].y) + (lhs[7] * rhs[1].x))));
}
#endif

__device__ inline qCudaCmplx zpow_real(const qCudaCmplx z, const qCudaReal1 p) {
  const qCudaReal1_f r = hypot((qCudaReal1_f)z.x, (qCudaReal1_f)z.y);
  const qCudaReal1_f theta = atan2((qCudaReal1_f)z.y, (qCudaReal1_f)z.x);

  const qCudaReal1_f rp = pow(r, (qCudaReal1_f)p);
  const qCudaReal1_f pt = p * theta;

  return ((qCudaReal1)rp) *
         sin((qCudaCmplx)((qCudaReal1)(pt + SineShift), (qCudaReal1)pt));
}

__device__ inline qCudaCmplx zexp(const qCudaCmplx z) {
  return ((qCudaReal1)exp((qCudaReal1_f)z.x)) *
         sin((qCudaCmplx)(z.y + SineShift, z.y));
}

__device__ inline qCudaCmplx zlog(const qCudaCmplx z) {
  const qCudaReal1_f r = hypot((qCudaReal1_f)z.x, (qCudaReal1_f)z.y);
  const qCudaReal1_f theta = atan2((qCudaReal1_f)z.y, (qCudaReal1_f)z.x);

  return (qCudaCmplx)((qCudaReal1)log(r), (qCudaReal1)theta);
}

__device__ inline qCudaReal1 arg(const qCudaCmplx cmp) {
  if (cmp.x == ZERO_R1 && cmp.y == ZERO_R1)
    return ZERO_R1;
  return (qCudaReal1)atan2((qCudaReal1_f)cmp.y, (qCudaReal1_f)cmp.x);
}

__device__ inline qCudaCmplx conj(const qCudaCmplx cmp) {
  return (qCudaCmplx)(cmp.x, -cmp.y);
}

__device__ inline qCudaCmplx polar_unit(const qCudaReal1 theta) {
  return sin((qCudaCmplx)(theta + (PI_R1 / 2), theta));
}

__device__ inline qCudaReal1 qCudaArg(const qCudaCmplx cmp) {
  if (cmp.x == (qCudaReal1)0.0f && cmp.y == (qCudaReal1)0.0f)
    return (qCudaReal1)0.0f;
  return (qCudaReal1)atan2((qCudaReal1_f)cmp.y, (qCudaReal1_f)cmp.x);
}

__device__ inline qCudaReal1 qCudaDot(qCudaReal2 a, qCudaReal2 b) {
  return a.x * b.x + a.y * b.y;
}

__device__ inline qCudaReal1 qCudaDot(qCudaReal4 a, qCudaReal4 b) {
#if FPPOW > 4
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
#else
  return a[0].x * b[0].x + a[0].y * b[0].y + a[1].x * b[1].x + a[1].y * b[1].y;
#endif
}

#if FPPOW > 4
__device__ inline qCudaCmplx polar_unit(const qCudaReal1 theta) {
  return make_qCudaCmplx(cos(theta), sin(theta));
}
#else
__device__ inline qCudaCmplx polar_unit(const qCudaReal1 theta) {
  return make_qCudaCmplx((qCudaReal1)cos((qCudaReal1_f)theta),
                         (qCudaReal1)sin((qCudaReal1_f)theta));
}
#endif

__device__ inline qCudaCmplx qCudaConj(qCudaCmplx a) {
  return make_qCudaCmplx(a.x, -a.y);
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
#define K vecCapIntArgs[9U]
#define l_X get_group_id(0)
#define l_Y get_group_id(1)
#define l_Z get_group_id(2)

__global__ void clear_buffer_int(symint *a) { a[i_X] = 0; }
__global__ void clear_buffer_real(qCudaReal1 *a) { a[i_X] = ZERO_R1; }
__global__ void fill_ones_int(symint *a) { a[i_X] = 1; }
__global__ void fill_ones_real(qCudaReal1 *a) { a[i_X] = ONE_R1; }
__global__ void fill_ones_complex(qCudaCmplx *a) {
  a[i_X] = (qCudaCmplx)(ONE_R1, ZERO_R1);
}
__global__ void fill_value_int(symint *a, symint *v) { a[i_X] = *v; }
__global__ void fill_value_real(qCudaReal1 *a, qCudaReal1 *v) { a[i_X] = *v; }
__global__ void fill_value_complex(qCudaCmplx *a, qCudaCmplx *v) {
  a[i_X] = *v;
}
__global__ void real_to_complex_buffer(qCudaReal1 *a, qCudaCmplx *b) {
  b[i_X] = (qCudaCmplx)(a[i_X], ZERO_R1);
}

__global__ void reduce_real(qCudaReal1 *a, qCudaReal1 *b, tcapint *shape,
                            tcapint *stride, tcapint *vecCapIntArgs) {
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

  qCudaReal1 sum = ZERO_R1;
  for (tcapint j = 0U; j < shape[id]; ++j) {
    sum += a[base + j * stride[id]];
  }
  b[i_X * I_B + O_B] = sum;
}
__global__ void reduce_complex(qCudaCmplx *a, qCudaCmplx *b, tcapint *shape,
                               tcapint *stride, tcapint *vecCapIntArgs) {
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

  qCudaCmplx sum = ZERO_R1;
  for (tcapint j = 0U; j < shape[id]; ++j) {
    sum += a[base + j * stride[id]];
  }
  b[i_X * I_B + O_B] = sum;
}
__global__ void reduce_grad_real(qCudaReal1 *din, qCudaReal1 *dout,
                                 tcapint *shape, tcapint *stride,
                                 tcapint *vecCapIntArgs) {
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
__global__ void reduce_grad_complex(qCudaCmplx *din, qCudaCmplx *dout,
                                    tcapint *shape, tcapint *stride,
                                    tcapint *vecCapIntArgs) {
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
__global__ void reduce_grad_mixed(qCudaReal1 *din, qCudaReal1 *dout,
                                  tcapint *shape, tcapint *stride,
                                  tcapint *vecCapIntArgs) {
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

__global__ void relu(qCudaReal1 *a, qCudaReal1 *out, tcapint *vecCapIntArgs) {
  const qCudaReal1 tmp = a[i_X * I_A + O_A];
  out[i_X * I_B + O_B] = (tmp > 0) ? tmp : ZERO_R1;
}
__global__ void relu_grad_real(qCudaReal1 *din, qCudaReal1 *in,
                               qCudaReal1 *dout, tcapint *vecCapIntArgs) {
  if (in[i_X * I_B + O_B] > ZERO_R1) {
    din[i_X * I_A + O_A] += dout[i_X * I_C + O_C];
  }
}
__global__ void relu_grad_complex(qCudaCmplx *din, qCudaReal1 *in,
                                  qCudaCmplx *dout, tcapint *vecCapIntArgs) {
  if (in[i_X * I_B + O_B] > ZERO_R1) {
    din[i_X * I_A + O_A] += dout[i_X * I_C + O_C];
  }
}
__global__ void relu_grad_mixed(qCudaReal1 *din, qCudaReal1 *in,
                                qCudaReal1 *dout, tcapint *vecCapIntArgs) {
  if (in[i_X * I_B + O_B] > 0) {
    din[(i_X * I_A + O_A) << 1U] += dout[i_X * I_C + O_C];
  }
}

__global__ void sigmoid(qCudaReal1 *a, qCudaReal1 *out,
                        tcapint *vecCapIntArgs) {
  out[i_X * I_B + O_B] =
      ONE_R1 / (ONE_R1 + (real1)exp((qCudaReal1_f)(-a[i_X * I_A + O_A])));
}
__global__ void sigmoid_grad_real(qCudaReal1 *din, qCudaReal1 *in,
                                  qCudaReal1 *dout, tcapint *vecCapIntArgs) {
  const qCudaReal1 yi = in[i_X * I_B + O_B];
  din[i_X * I_A + O_A] += yi * (ONE_R1 - yi) * dout[i_X * I_C + O_C];
}
__global__ void sigmoid_grad_complex(qCudaCmplx *din, qCudaReal1 *in,
                                     qCudaCmplx *dout, tcapint *vecCapIntArgs) {
  const qCudaReal1 yi = in[i_X * I_B + O_B];
  din[i_X * I_A + O_A] += yi * (ONE_R1 - yi) * dout[i_X * I_C + O_C];
}
__global__ void sigmoid_grad_mixed(qCudaReal1 *din, qCudaReal1 *in,
                                   qCudaReal1 *dout, tcapint *vecCapIntArgs) {
  const qCudaReal1 yi = in[i_X * I_B + O_B];
  din[(i_X * I_A + O_A) << 1U] += yi * (ONE_R1 - yi) * dout[i_X * I_C + O_C];
}

__global__ void wtanh(qCudaReal1 *a, qCudaReal1 *out, tcapint *vecCapIntArgs) {
  out[i_X * I_B + O_B] = tanh(a[i_X * I_A + O_A]);
}
__global__ void wtanh_grad_real(qCudaReal1 *din, qCudaReal1 *in,
                                qCudaReal1 *dout, tcapint *vecCapIntArgs) {
  const qCudaReal1 yi = in[i_X * I_B + O_B];
  din[i_X * I_A + O_A] += (ONE_R1 - yi * yi) * dout[i_X * I_C + O_C];
}
__global__ void wtanh_grad_complex(qCudaCmplx *din, qCudaReal1 *in,
                                   qCudaCmplx *dout, tcapint *vecCapIntArgs) {
  const qCudaReal1 yi = in[i_X * I_B + O_B];
  din[i_X * I_A + O_A] += (ONE_R1 - yi * yi) * dout[i_X * I_C + O_C];
}
__global__ void wtanh_grad_mixed(qCudaReal1 *din, qCudaReal1 *in,
                                 qCudaReal1 *dout, tcapint *vecCapIntArgs) {
  const qCudaReal1 yi = in[i_X * I_B + O_B];
  din[(i_X * I_A + O_A) << 1U] += (ONE_R1 - yi * yi) * dout[i_X * I_C + O_C];
}

__global__ void match_grad_real(qCudaReal1 *din, qCudaReal1 *in,
                                qCudaReal1 *dout, tcapint *vecCapIntArgs,
                                qCudaReal1 *m) {
  if (*m == in[i_X * I_B + O_B]) {
    din[i_X * I_A + O_A] += dout[i_X * I_C + O_C];
  }
}
__global__ void match_grad_complex(qCudaCmplx *din, qCudaReal1 *in,
                                   qCudaCmplx *dout, tcapint *vecCapIntArgs,
                                   qCudaReal1 *m) {
  if (*m == in[i_X * I_B + O_B]) {
    din[i_X * I_A + O_A] += dout[i_X * I_C + O_C];
  }
}
__global__ void match_grad_mixed(qCudaReal1 *din, qCudaReal1 *in,
                                 qCudaReal1 *dout, tcapint *vecCapIntArgs,
                                 qCudaReal1 *m) {
  if (*m == in[i_X * I_B + O_B]) {
    din[(i_X * I_A + O_A) << 1U] += dout[i_X * I_C + O_C];
  }
}

__global__ void clamp_real(qCudaReal1 *a, qCudaReal1 *out,
                           tcapint *vecCapIntArgs, qCudaCmplx *p) {
  const qCudaReal1 tmp = a[i_X * I_A + O_A];
  const qCudaCmplx b = *p;
  out[i_X * I_B + O_B] = (tmp > b.x) ? ((tmp < b.y) ? tmp : b.y) : b.x;
}
__global__ void clamp_grad_real(qCudaReal1 *din, qCudaReal1 *in,
                                qCudaReal1 *dout, tcapint *vecCapIntArgs,
                                qCudaCmplx *p) {
  const qCudaReal1 xi = in[i_X * I_B + O_B];
  const qCudaCmplx b = *p;
  if (xi > b.x && xi < b.y) {
    din[i_X * I_C + O_C] += dout[i_X * I_A + O_A];
  }
}
__global__ void clamp_grad_complex(qCudaCmplx *din, qCudaReal1 *in,
                                   qCudaCmplx *dout, tcapint *vecCapIntArgs,
                                   qCudaCmplx *p) {
  const qCudaReal1 xi = in[i_X * I_B + O_B];
  const qCudaCmplx b = *p;
  if (xi > b.x && xi < b.y) {
    din[i_X * I_C + O_C] += dout[i_X * I_A + O_A];
  }
}
__global__ void clamp_grad_mixed(qCudaReal1 *din, qCudaReal1 *in,
                                 qCudaReal1 *dout, tcapint *vecCapIntArgs,
                                 qCudaCmplx *p) {
  const qCudaReal1 xi = in[i_X * I_B + O_B];
  const qCudaCmplx b = *p;
  if (xi > b.x && xi < b.y) {
    din[(i_X * I_C + O_C) << 1U] += dout[i_X * I_A + O_A];
  }
}

__global__ void abs_real(qCudaReal1 *a, qCudaReal1 *out,
                         tcapint *vecCapIntArgs) {
  const qCudaReal1 tmp = a[i_X * I_A + O_A];
  out[i_X * I_B + O_B] = (tmp < 0) ? -tmp : tmp;
}
__global__ void abs_complex(qCudaCmplx *a, qCudaReal1 *out,
                            tcapint *vecCapIntArgs) {
  const qCudaCmplx tmp = a[i_X * I_A + O_A];
  out[i_X * I_B + O_B] = sqrt(qCudaDot(tmp, tmp));
}
__global__ void abs_real_grad_real(qCudaReal1 *din, qCudaReal1 *in,
                                   qCudaReal1 *dout, tcapint *vecCapIntArgs) {
  const qCudaReal1 tmp = in[i_X * I_B + O_B];
  if (tmp != ZERO_R1) {
    const qCudaReal1 tmp_o = dout[i_X * I_C + O_C];
    din[i_X * I_A + O_A] += (tmp > ZERO_R1) ? tmp_o : -tmp_o;
  }
}
__global__ void abs_real_grad_complex(qCudaCmplx *din, qCudaReal1 *in,
                                      qCudaCmplx *dout,
                                      tcapint *vecCapIntArgs) {
  const qCudaReal1 tmp = in[i_X * I_B + O_B];
  if (tmp != ZERO_R1) {
    const qCudaCmplx tmp_o = dout[i_X * I_C + O_C];
    din[i_X * I_A + O_A] += (tmp > ZERO_R1) ? tmp_o : -tmp_o;
  }
}
__global__ void abs_real_grad_mixed(qCudaReal1 *din, qCudaReal1 *in,
                                    qCudaReal1 *dout, tcapint *vecCapIntArgs) {
  const qCudaReal1 tmp = in[i_X * I_B + O_B];
  if (tmp != ZERO_R1) {
    const qCudaReal1 tmp_o = dout[i_X * I_C + O_C];
    din[(i_X * I_A + O_A) << 1U] += (tmp > ZERO_R1) ? tmp_o : -tmp_o;
  }
}
__global__ void abs_complex_grad_real(qCudaCmplx *din, qCudaCmplx *in,
                                      qCudaReal1 *dout,
                                      tcapint *vecCapIntArgs) {
  const qCudaCmplx tmp = in[i_X * I_B + O_B];
  if ((tmp.x != ZERO_R1) || (tmp.y != ZERO_R1)) {
    din[i_X * I_A + O_A] +=
        (dout[i_X * I_C + O_C] / sqrt(qCudaDot(tmp, tmp))) * tmp;
  }
}
__global__ void abs_complex_grad_complex(qCudaCmplx *din, qCudaCmplx *in,
                                         qCudaCmplx *dout,
                                         tcapint *vecCapIntArgs) {
  const qCudaCmplx tmp = in[i_X * I_B + O_B];
  if ((tmp.x != ZERO_R1) || (tmp.y != ZERO_R1)) {
    din[i_X * I_A + O_A] +=
        zmul(dout[i_X * I_C + O_C], tmp) / sqrt(qCudaDot(tmp, tmp));
  }
}
__global__ void abs_complex_grad_mixed(qCudaCmplx *din, qCudaCmplx *in,
                                       qCudaReal1 *dout,
                                       tcapint *vecCapIntArgs) {
  const qCudaCmplx tmp = in[i_X * I_B + O_B];
  if ((tmp.x != ZERO_R1) || (tmp.y != ZERO_R1)) {
    din[i_X * I_A + O_A] +=
        (dout[i_X * I_C + O_C] / sqrt(qCudaDot(tmp, tmp))) * tmp;
  }
}

__global__ void add_real(qCudaReal1 *a, qCudaReal1 *b, qCudaReal1 *out,
                         tcapint *vecCapIntArgs) {
  out[i_X * I_C + O_C] = a[i_X * I_A + O_A] + b[i_X * I_B + O_B];
}
__global__ void add_complex(qCudaCmplx *a, qCudaCmplx *b, qCudaCmplx *out,
                            tcapint *vecCapIntArgs) {
  out[i_X * I_C + O_C] = a[i_X * I_A + O_A] + b[i_X * I_B + O_B];
}
__global__ void add_mixed(qCudaCmplx *a, qCudaReal1 *b, qCudaCmplx *out,
                          tcapint *vecCapIntArgs) {
  out[i_X * I_C + O_C] =
      a[i_X * I_A + O_A] + (qCudaCmplx)(b[i_X * I_B + O_B], 0);
}

__global__ void mul_real(qCudaReal1 *a, qCudaReal1 *b, qCudaReal1 *out,
                         tcapint *vecCapIntArgs) {
  out[i_X * I_C + O_C] = a[i_X * I_A + O_A] * b[i_X * I_B + O_B];
}
__global__ void mul_complex(qCudaCmplx *a, qCudaCmplx *b, qCudaCmplx *out,
                            tcapint *vecCapIntArgs) {
  out[i_X * I_C + O_C] = zmul(a[i_X * I_A + O_A], b[i_X * I_B + O_B]);
}
__global__ void mul_mixed(qCudaCmplx *a, qCudaReal1 *b, qCudaCmplx *out,
                          tcapint *vecCapIntArgs) {
  out[i_X * I_C + O_C] = b[i_X * I_B + O_B] * a[i_X * I_A + O_A];
}

__global__ void matmul_real(qCudaReal1 *a, qCudaReal1 *b, qCudaReal1 *out,
                            tcapint *vecCapIntArgs) {
  qCudaReal1 sum = ZERO_R1;
  for (tcapint k = 0; k < K; ++k) {
    const tcapint a_idx = (O_A + i_X * I_A + k * J_A);
    const tcapint b_idx = (O_B + k * I_B + i_Y * J_B);
    sum += a[a_idx] * b[b_idx];
  }
  const tcapint o_idx = O_C + i_X * I_C + i_Y * J_C;
  out[o_idx] = sum;
}
__global__ void matmul_complex(qCudaCmplx *a, qCudaCmplx *b, qCudaCmplx *out,
                               tcapint *vecCapIntArgs) {
  qCudaCmplx sum = ZERO_R1;
  for (tcapint k = 0; k < K; ++k) {
    const tcapint a_idx = (O_A + i_X * I_A + k * J_A);
    const tcapint b_idx = (O_B + k * I_B + i_Y * J_B);
    sum += zmul(a[a_idx], b[b_idx]);
  }
  const tcapint o_idx = O_C + i_X * I_C + i_Y * J_C;
  out[o_idx] = sum;
}
__global__ void matmul_mixed_c_left(qCudaCmplx *a, qCudaReal1 *b,
                                    qCudaCmplx *out, tcapint *vecCapIntArgs) {
  qCudaCmplx sum = ZERO_R1;
  for (tcapint k = 0; k < K; ++k) {
    const tcapint a_idx = (O_A + i_X * I_A + k * J_A);
    const tcapint b_idx = (O_B + k * I_B + i_Y * J_B);
    sum += b[b_idx] * a[a_idx];
  }
  const tcapint o_idx = O_C + i_X * I_C + i_Y * J_C;
  out[o_idx] = sum;
}
__global__ void matmul_mixed_c_right(qCudaReal1 *a, qCudaCmplx *b,
                                     qCudaCmplx *out, tcapint *vecCapIntArgs) {
  qCudaCmplx sum = ZERO_R1;
  for (tcapint k = 0; k < K; ++k) {
    const tcapint a_idx = (O_A + i_X * I_A + k * J_A);
    const tcapint b_idx = (O_B + k * I_B + i_Y * J_B);
    sum += a[a_idx] * b[b_idx];
  }
  const tcapint o_idx = O_C + i_X * I_C + i_Y * J_C;
  out[o_idx] = sum;
}

__global__ void sub_real(qCudaReal1 *a, qCudaReal1 *b, qCudaReal1 *out,
                         tcapint *vecCapIntArgs) {
  out[i_X * I_C + O_C] = a[i_X * I_A + O_A] - b[i_X * I_B + O_B];
}
__global__ void sub_complex(qCudaCmplx *a, qCudaCmplx *b, qCudaCmplx *out,
                            tcapint *vecCapIntArgs) {
  out[i_X * I_C + O_C] = a[i_X * I_A + O_A] - b[i_X * I_B + O_B];
}
__global__ void sub_mixed_c_left(qCudaCmplx *a, qCudaReal1 *b, qCudaCmplx *out,
                                 tcapint *vecCapIntArgs) {
  out[i_X * I_C + O_C] =
      a[i_X * I_A + O_A] - (qCudaCmplx)(b[i_X * I_B + O_B], ZERO_R1);
}
__global__ void sub_mixed_c_right(qCudaReal1 *a, qCudaCmplx *b, qCudaCmplx *out,
                                  tcapint *vecCapIntArgs) {
  out[i_X * I_C + O_C] =
      (qCudaCmplx)(a[i_X * I_A + O_A], ZERO_R1) - b[i_X * I_B + O_B];
}

__global__ void div_real(qCudaReal1 *a, qCudaReal1 *b, qCudaReal1 *out,
                         tcapint *vecCapIntArgs) {
  out[i_X * I_C + O_C] = a[i_X * I_A + O_A] / b[i_X * I_B + O_B];
}
__global__ void div_complex(qCudaCmplx *a, qCudaCmplx *b, qCudaCmplx *out,
                            tcapint *vecCapIntArgs) {
  out[i_X * I_C + O_C] = zdiv(a[i_X * I_A + O_A], b[i_X * I_B + O_B]);
}
__global__ void div_mixed_c_left(qCudaCmplx *a, qCudaReal1 *b, qCudaCmplx *out,
                                 tcapint *vecCapIntArgs) {
  out[i_X * I_C + O_C] = a[i_X * I_A + O_A] / b[i_X * I_B + O_B];
}
__global__ void div_mixed_c_right(qCudaReal1 *a, qCudaCmplx *b, qCudaCmplx *out,
                                  tcapint *vecCapIntArgs) {
  out[i_X * I_C + O_C] =
      zdiv((qCudaCmplx)(a[i_X * I_A + O_A], ZERO_R1), b[i_X * I_B + O_B]);
}

__global__ void add_in_place_real(qCudaReal1 *a, qCudaReal1 *b,
                                  tcapint *vecCapIntArgs) {
  a[i_X * I_A + O_A] += b[i_X * I_B + O_B];
}
__global__ void add_in_place_complex(qCudaCmplx *a, qCudaCmplx *b,
                                     tcapint *vecCapIntArgs) {
  a[i_X * I_A + O_A] += b[i_X * I_B + O_B];
}
__global__ void add_in_place_mixed(qCudaCmplx *a, qCudaReal1 *b,
                                   tcapint *vecCapIntArgs) {
  a[i_X * I_A + O_A] += (qCudaCmplx)(b[i_X * I_B + O_B], 0);
}

__global__ void sub_in_place_real(qCudaReal1 *a, qCudaReal1 *b,
                                  tcapint *vecCapIntArgs) {
  a[i_X * I_A + O_A] -= b[i_X * I_B + O_B];
}
__global__ void sub_in_place_complex(qCudaCmplx *a, qCudaCmplx *b,
                                     tcapint *vecCapIntArgs) {
  a[i_X * I_A + O_A] -= b[i_X * I_B + O_B];
}
__global__ void sub_in_place_mixed(qCudaCmplx *a, qCudaReal1 *b,
                                   tcapint *vecCapIntArgs) {
  a[i_X * I_A + O_A] -= (qCudaCmplx)(b[i_X * I_B + O_B], 0);
}

__global__ void pow_real(qCudaReal1 *a, qCudaReal1 *out, tcapint *vecCapIntArgs,
                         qCudaReal1 *p) {
  out[i_X * I_B + O_B] =
      (real1)pow((qCudaReal1_f)a[i_X * I_A + O_A], (qCudaReal1_f)*p);
}
__global__ void pow_complex(qCudaCmplx *a, qCudaCmplx *out,
                            tcapint *vecCapIntArgs, qCudaReal1 *p) {
  out[i_X * I_B + O_B] = zpow_real(a[i_X * I_A + O_A], *p);
}
__global__ void exp_real(qCudaReal1 *a, qCudaReal1 *out, tcapint *vecCapIntArgs,
                         qCudaReal1 *log_b) {
  out[i_X * I_B + O_B] =
      ((real1)exp((qCudaReal1_f)(a[i_X * I_A + O_A]) * (*log_b)));
}
__global__ void exp_complex(qCudaCmplx *a, qCudaCmplx *out,
                            tcapint *vecCapIntArgs, qCudaReal1 *log_b) {
  out[i_X * I_B + O_B] = zexp(a[i_X * I_A + O_A] * (*log_b));
}
__global__ void log_real(qCudaReal1 *a, qCudaReal1 *out, tcapint *vecCapIntArgs,
                         qCudaReal1 *inv_log_b) {
  out[i_X * I_B + O_B] =
      ((real1)log((qCudaReal1_f)a[i_X * I_A + O_A])) * (*inv_log_b);
}
__global__ void log_complex(qCudaCmplx *a, qCudaCmplx *out,
                            tcapint *vecCapIntArgs, qCudaReal1 *inv_log_b) {
  out[i_X * I_B + O_B] = zlog(a[i_X * I_A + O_A]) * (*inv_log_b);
}

__global__ void embedding_real(symint *idx, qCudaReal1 *W, qCudaReal1 *O,
                               tcapint *vecCapIntArgs) {
  const tcapint token = idx[O_A + i_X * I_A];
  const tcapint w_base = O_B + token * I_B;
  const tcapint o_base = J_A + i_X * O_C;
  for (tcapint d = 0U; d < J_B; ++d) {
    O[o_base + d * O_C] = W[w_base + d * I_C];
  }
}
__global__ void embedding_complex(symint *idx, qCudaCmplx *W, qCudaCmplx *O,
                                  tcapint *vecCapIntArgs) {
  const tcapint token = idx[O_A + i_X * I_A];
  const tcapint w_base = O_B + token * I_B;
  const tcapint o_base = J_A + i_X * O_C;
  for (tcapint d = 0U; d < J_B; ++d) {
    O[o_base + d * O_C] = W[w_base + d * I_C];
  }
}
__global__ void embedding_grad_real(symint *idx, qCudaReal1 *dW, qCudaReal1 *dO,
                                    tcapint *vecCapIntArgs) {
  const tcapint token = idx[O_A + i_X * I_A];
  const tcapint w_base = O_B + token * I_B;
  const tcapint o_base = J_A + i_X * O_C;
  for (tcapint d = 0U; d < J_B; ++d) {
    dW[w_base + d * O_C] += dO[o_base + d * I_C];
  }
}
__global__ void embedding_grad_complex(symint *idx, qCudaCmplx *dW,
                                       qCudaCmplx *dO, tcapint *vecCapIntArgs) {
  const tcapint token = idx[O_A + i_X * I_A];
  const tcapint w_base = O_B + token * I_B;
  const tcapint o_base = J_A + i_X * O_C;
  for (tcapint d = 0U; d < J_B; ++d) {
    dW[w_base + d * O_C] += dO[o_base + d * I_C];
  }
}
__global__ void embedding_grad_mixed(symint *idx, qCudaReal1 *dW,
                                     qCudaReal1 *dO, tcapint *vecCapIntArgs) {
  const tcapint token = idx[O_A + i_X * I_A];
  const tcapint w_base = O_B + token * I_B;
  const tcapint o_base = J_A + i_X * O_C;
  for (tcapint d = 0U; d < J_B; ++d) {
    dW[(w_base + d * O_C) << 1U] += dO[o_base + d * I_C];
  }
}

__global__ void copy_real(qCudaReal1 *a, qCudaReal1 *b,
                          tcapint *vecCapIntArgs) {
  a[i_X * I_A + O_A] = b[i_X * I_B + O_B];
}
__global__ void copy_complex(qCudaCmplx *a, qCudaCmplx *b,
                             tcapint *vecCapIntArgs) {
  a[i_X * I_A + O_A] = b[i_X * I_B + O_B];
}
} // namespace Weed
