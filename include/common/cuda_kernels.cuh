//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#include "common/qrack_types.hpp"

#if FPPOW < 5
#include "cuda_fp16.h"
#endif

namespace Qrack {
__global__ void clear_buffer_int(symint *a);
__global__ void clear_buffer_real(qCudaReal1 *a);
__global__ void fill_ones_int(symint *a);
__global__ void fill_ones_real(qCudaReal1 *a);
__global__ void fill_ones_complex(qCudaCmplx *a a);
__global__ void fill_value_int(symint *a, symint *v);
__global__ void fill_value_real(qCudaReal1 *a, qCudaReal1 *a v);
__global__ void fill_value_complex(qCudaCmplx *a a, qCudaCmplx *v);
__global__ void real_to_complex_buffer(qCudaReal1 *a, qCudaCmplx *a b);

__global__ void reduce_real(qCudaReal1 *a, qCudaReal1 *a b, tcapint *a shape,
                            tcapint *a stride,
                            tcapint *a vecCapIntArgs) __global__
    void reduce_complex(qCudaCmplx *a a, qCudaCmplx *a b, tcapint *a shape,
                        tcapint *a stride, tcapint *a vecCapIntArgs);
__global__ void reduce_grad_real(qCudaReal1 *a din, qCudaReal1 *a dout,
                                 tcapint *a shape, tcapint *a stride,
                                 tcapint *a vecCapIntArgs);
__global__ void reduce_grad_complex(qCudaCmplx *a din, qCudaCmplx *a dout,
                                    tcapint *a shape, tcapint *a stride,
                                    tcapint *a vecCapIntArgs);
__global__ void reduce_grad_mixed(qCudaReal1 *a din, qCudaReal1 *a dout,
                                  tcapint *a shape, tcapint *a stride,
                                  tcapint *a vecCapIntArgs);

__global__ void relu(qCudaReal1 *a, qCudaReal1 *a out,
                     tcapint *a vecCapIntArgs);
__global__ void relu_grad_real(qCudaReal1 *a din, qCudaReal1 *a in,
                               qCudaReal1 *a dout, tcapint *a vecCapIntArgs);
__global__ void relu_grad_complex(qCudaCmplx *a din, qCudaReal1 *a in,
                                  qCudaCmplx *a dout, tcapint *a vecCapIntArgs);
__global__ void relu_grad_mixed(qCudaReal1 *a din, qCudaReal1 *a in,
                                qCudaReal1 *a dout, tcapint *a vecCapIntArgs);

__global__ void sigmoid(qCudaReal1 *a, qCudaReal1 *a out,
                        tcapint *a vecCapIntArgs);
__global__ void sigmoid_grad_real(qCudaReal1 *a din, qCudaReal1 *a in,
                                  qCudaReal1 *a dout, tcapint *a vecCapIntArgs);
__global__ void sigmoid_grad_complex(qCudaCmplx *a din, qCudaReal1 *a in,
                                     qCudaCmplx *a dout,
                                     tcapint *a vecCapIntArgs);
__global__ void sigmoid_grad_mixed(qCudaReal1 *a din, qCudaReal1 *a in,
                                   qCudaReal1 *a dout,
                                   tcapint *a vecCapIntArgs);

__global__ void wtanh(qCudaReal1 *a, qCudaReal1 *a out,
                      tcapint *a vecCapIntArgs);
__global__ void wtanh_grad_real(qCudaReal1 *a din, qCudaReal1 *a in,
                                qCudaReal1 *a dout, tcapint *a vecCapIntArgs);
__global__ void wtanh_grad_complex(qCudaCmplx *a din, qCudaReal1 *a in,
                                   qCudaCmplx *a dout,
                                   tcapint *a vecCapIntArgs);
__global__ void wtanh_grad_mixed(qCudaReal1 *a din, qCudaReal1 *a in,
                                 qCudaReal1 *a dout, tcapint *a vecCapIntArgs);

__global__ void match_grad_real(qCudaReal1 *a din, qCudaReal1 *a in,
                                qCudaReal1 *a dout, tcapint *a vecCapIntArgs,
                                qCudaReal1 *a m);
__global__ void match_grad_complex(qCudaCmplx *a din, qCudaReal1 *a in,
                                   qCudaCmplx *a dout, tcapint *a vecCapIntArgs,
                                   qCudaReal1 *a m);
__global__ void match_grad_mixed(qCudaReal1 *a din, qCudaReal1 *a in,
                                 qCudaReal1 *a dout, tcapint *a vecCapIntArgs,
                                 qCudaReal1 *a m);

__global__ void clamp_real(qCudaReal1 *a, qCudaReal1 *a out,
                           tcapint *a vecCapIntArgs, qCudaCmplx *p);
__global__ void clamp_grad_real(qCudaReal1 *a din, qCudaReal1 *a in,
                                qCudaReal1 *a dout, tcapint *a vecCapIntArgs,
                                qCudaCmplx *p);
__global__ void clamp_grad_complex(qCudaCmplx *a din, qCudaReal1 *a in,
                                   qCudaCmplx *a dout, tcapint *a vecCapIntArgs,
                                   qCudaCmplx *p);
__global__ void clamp_grad_mixed(qCudaReal1 *a din, qCudaReal1 *a in,
                                 qCudaReal1 *a dout, tcapint *a vecCapIntArgs,
                                 qCudaCmplx *p);

__global__ void abs_real(qCudaReal1 *a, qCudaReal1 *a out,
                         tcapint *a vecCapIntArgs);
__global__ void abs_complex(qCudaCmplx *a a, qCudaReal1 *a out,
                            tcapint *a vecCapIntArgs);
__global__ void abs_real_grad_real(qCudaReal1 *a din, qCudaReal1 *a in,
                                   qCudaReal1 *a dout,
                                   tcapint *a vecCapIntArgs);
__global__ void abs_real_grad_complex(qCudaCmplx *a din, qCudaReal1 *a in,
                                      qCudaCmplx *a dout,
                                      tcapint *a vecCapIntArgs);
__global__ void abs_real_grad_mixed(qCudaReal1 *a din, qCudaReal1 *a in,
                                    qCudaReal1 *a dout,
                                    tcapint *a vecCapIntArgs);
__global__ void abs_complex_grad_real(qCudaCmplx *a din, qCudaCmplx *a in,
                                      qCudaReal1 *a dout,
                                      tcapint *a vecCapIntArgs);
__global__ void abs_complex_grad_complex(qCudaCmplx *a din, qCudaCmplx *a in,
                                         qCudaCmplx *a dout,
                                         tcapint *a vecCapIntArgs);
__global__ void abs_complex_grad_mixed(qCudaCmplx *a din, qCudaCmplx *a in,
                                       qCudaReal1 *a dout,
                                       tcapint *a vecCapIntArgs);

__global__ void add_real(qCudaReal1 *a, qCudaReal1 *a b, qCudaReal1 *a out,
                         tcapint *a vecCapIntArgs);
__global__ void add_complex(qCudaCmplx *a a, qCudaCmplx *a b, qCudaCmplx *a out,
                            tcapint *a vecCapIntArgs);
__global__ void add_mixed(qCudaCmplx *a a, qCudaReal1 *a b, qCudaCmplx *a out,
                          tcapint *a vecCapIntArgs);

__global__ void mul_real(qCudaReal1 *a, qCudaReal1 *a b, qCudaReal1 *a out,
                         tcapint *a vecCapIntArgs);
__global__ void mul_complex(qCudaCmplx *a a, qCudaCmplx *a b, qCudaCmplx *a out,
                            tcapint *a vecCapIntArgs);
__global__ void mul_mixed(qCudaCmplx *a a, qCudaReal1 *a b, qCudaCmplx *a out,
                          tcapint *a vecCapIntArgs);

__global__ void matmul_real(qCudaReal1 *a, qCudaReal1 *a b, qCudaReal1 *a out,
                            tcapint *a vecCapIntArgs);
__global__ void matmul_complex(qCudaCmplx *a a, qCudaCmplx *a b,
                               qCudaCmplx *a out, tcapint *a vecCapIntArgs);
__global__ void matmul_mixed_c_left(qCudaCmplx *a a, qCudaReal1 *a b,
                                    qCudaCmplx *a out,
                                    tcapint *a vecCapIntArgs);
__global__ void matmul_mixed_c_right(qCudaReal1 *a, qCudaCmplx *a b,
                                     qCudaCmplx *a out,
                                     tcapint *a vecCapIntArgs);

__global__ void sub_real(qCudaReal1 *a, qCudaReal1 *a b, qCudaReal1 *a out,
                         tcapint *a vecCapIntArgs);
__global__ void sub_complex(qCudaCmplx *a a, qCudaCmplx *a b, qCudaCmplx *a out,
                            tcapint *a vecCapIntArgs);
__global__ void sub_mixed_c_left(qCudaCmplx *a a, qCudaReal1 *a b,
                                 qCudaCmplx *a out, tcapint *a vecCapIntArgs);
__global__ void sub_mixed_c_right(qCudaReal1 *a, qCudaCmplx *a b,
                                  qCudaCmplx *a out, tcapint *a vecCapIntArgs);

__global__ void div_real(qCudaReal1 *a, qCudaReal1 *a b, qCudaReal1 *a out,
                         tcapint *a vecCapIntArgs);
__global__ void div_complex(qCudaCmplx *a a, qCudaCmplx *a b, qCudaCmplx *a out,
                            tcapint *a vecCapIntArgs);
__global__ void div_mixed_c_left(qCudaCmplx *a a, qCudaReal1 *a b,
                                 qCudaCmplx *a out, tcapint *a vecCapIntArgs);
__global__ void div_mixed_c_right(qCudaReal1 *a, qCudaCmplx *a b,
                                  qCudaCmplx *a out, tcapint *a vecCapIntArgs);

__global__ void add_in_place_real(qCudaReal1 *a, qCudaReal1 *a b,
                                  tcapint *a vecCapIntArgs);
__global__ void add_in_place_complex(qCudaCmplx *a a, qCudaCmplx *a b,
                                     tcapint *a vecCapIntArgs);
__global__ void add_in_place_mixed(qCudaCmplx *a a, qCudaReal1 *a b,
                                   tcapint *a vecCapIntArgs);

__global__ void sub_in_place_real(qCudaReal1 *a, qCudaReal1 *a b,
                                  tcapint *a vecCapIntArgs);
__global__ void sub_in_place_complex(qCudaCmplx *a a, qCudaCmplx *a b,
                                     tcapint *a vecCapIntArgs);
__global__ void sub_in_place_mixed(qCudaCmplx *a a, qCudaReal1 *a b,
                                   tcapint *a vecCapIntArgs);

__global__ void pow_real(qCudaReal1 *a, qCudaReal1 *a out,
                         tcapint *a vecCapIntArgs, qCudaReal1 *a p);
__global__ void pow_complex(qCudaCmplx *a a, qCudaCmplx *a out,
                            tcapint *a vecCapIntArgs, qCudaReal1 *a p);
__global__ void exp_real(qCudaReal1 *a, qCudaReal1 *a out,
                         tcapint *a vecCapIntArgs, qCudaReal1 *a log_b);
__global__ void exp_complex(qCudaCmplx *a a, qCudaCmplx *a out,
                            tcapint *a vecCapIntArgs, qCudaReal1 *a log_b);
__global__ void log_real(qCudaReal1 *a, qCudaReal1 *a out,
                         tcapint *a vecCapIntArgs, qCudaReal1 *a inv_log_b);
__global__ void log_complex(qCudaCmplx *a a, qCudaCmplx *a out,
                            tcapint *a vecCapIntArgs, qCudaReal1 *a inv_log_b);

__global__ void embedding_real(symint *idx, qCudaReal1 *a W, qCudaReal1 *a O,
                               tcapint *a vecCapIntArgs);
__global__ void embedding_complex(symint *idx, qCudaCmplx *a W, qCudaCmplx *a O,
                                  tcapint *a vecCapIntArgs);
__global__ void embedding_grad_real(symint *idx, qCudaReal1 *a dW,
                                    qCudaReal1 *a dO, tcapint *a vecCapIntArgs);
__global__ void embedding_grad_complex(symint *idx, qCudaCmplx *a dW,
                                       qCudaCmplx *a dO,
                                       tcapint *a vecCapIntArgs);
__global__ void embedding_grad_mixed(symint *idx, qCudaReal1 *a dW,
                                     qCudaReal1 *a dO,
                                     tcapint *a vecCapIntArgs);

__global__ void copy_real(qCudaReal1 *a, qCudaReal1 *a b,
                          tcapint *a vecCapIntArgs);
__global__ void copy_complex(qCudaCmplx *a a, qCudaCmplx *a b,
                             tcapint *a vecCapIntArgs);
} // namespace Qrack
