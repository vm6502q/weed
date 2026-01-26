//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.en.html for details.

#include <iostream>

#include "catch.hpp"

#include "tests.hpp"

#include "autograd/sgd.hpp"
#include "autograd/zero_grad.hpp"
#include "tensors/complex_scalar.hpp"
#include "tensors/real_scalar.hpp"

using namespace Weed;

#define R(v) real1(v)
#define C(v) complex(v)

#define EPSILON 0.01f
#define REQUIRE_FLOAT(A, B)                                                    \
  do {                                                                         \
    real1_f __tmp_a = A;                                                       \
    real1_f __tmp_b = B;                                                       \
    REQUIRE(__tmp_a < (__tmp_b + EPSILON));                                    \
    REQUIRE(__tmp_a > (__tmp_b - EPSILON));                                    \
  } while (0);
#define REQUIRE_CMPLX(A, B) REQUIRE(std::norm((A) - (B)) < EPSILON);

#define GET_REAL(ptr) static_cast<RealScalar *>((ptr).get())->get_item()
#define GET_COMPLEX(ptr) static_cast<ComplexScalar *>((ptr).get())->get_item()

TEST_CASE("test_sum_real") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{R(1), R(2), R(3)}, std::vector<vecCapInt>{3, 1},
      std::vector<vecCapInt>{1, 3}, true, TEST_DTAG);
  TensorPtr y = Tensor::sum(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(6));
  REQUIRE(GET_REAL(x->grad) == R(1));
}

TEST_CASE("test_sum_complex") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<complex>{R(1), R(2), R(3)}, std::vector<vecCapInt>{3, 1},
      std::vector<vecCapInt>{1, 3}, true, TEST_DTAG);
  TensorPtr y = Tensor::sum(x);
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), R(6));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1));
}

TEST_CASE("test_mean_real") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{R(1), R(2), R(3)}, std::vector<vecCapInt>{3, 1},
      std::vector<vecCapInt>{1, 3}, true, TEST_DTAG);
  TensorPtr y = Tensor::mean(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE(GET_REAL(x->grad) == R(R(1) / R(3)));
}

TEST_CASE("test_mean_complex") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<complex>{R(1), R(2), R(3)}, std::vector<vecCapInt>{3, 1},
      std::vector<vecCapInt>{1, 3}, true, TEST_DTAG);
  TensorPtr y = Tensor::mean(x);
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(R(1) / R(3)));
}

TEST_CASE("test_scalar_relu") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = Tensor::relu(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE(GET_REAL(x->grad) == R(1));

  x = std::make_shared<RealScalar>(R(-2), true, TEST_DTAG);
  y = Tensor::relu(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == ZERO_CMPLX);
  REQUIRE(GET_REAL(x->grad) == ZERO_CMPLX);
}

TEST_CASE("test_scalar_relu_complex_grad") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = Tensor::relu(x);
  TensorPtr z = std::make_shared<ComplexScalar>(R(2), true, TEST_DTAG);
  TensorPtr w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(2));

  x = std::make_shared<RealScalar>(R(-2), true, TEST_DTAG);
  y = Tensor::relu(x);
  z = std::make_shared<ComplexScalar>(R(2), true, TEST_DTAG);
  w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == ZERO_CMPLX);
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), ZERO_CMPLX);
}

TEST_CASE("test_scalar_sigmoid") {
  TensorPtr x = std::make_shared<RealScalar>(ZERO_R1, true, TEST_DTAG);
  TensorPtr y = Tensor::sigmoid(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == ZERO_R1);
  REQUIRE(GET_REAL(x->grad) == ZERO_R1);
}

TEST_CASE("test_scalar_sigmoid_complex_grad") {
  TensorPtr x = std::make_shared<RealScalar>(R(0), true, TEST_DTAG);
  TensorPtr y = Tensor::sigmoid(x);
  TensorPtr z = std::make_shared<ComplexScalar>(R(1), true, TEST_DTAG);
  TensorPtr w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == ZERO_R1);
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), ZERO_R1);
}

TEST_CASE("test_scalar_clamp") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = Tensor::clamp(x, R(1), R(3));
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE(GET_REAL(x->grad) == R(1));

  x = std::make_shared<RealScalar>(ZERO_R1, true, TEST_DTAG);
  y = Tensor::clamp(x, R(1), R(3));
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(1));
  REQUIRE(GET_REAL(x->grad) == ZERO_R1);

  x = std::make_shared<RealScalar>(R(4), true, TEST_DTAG);
  y = Tensor::clamp(x, R(1), R(3));
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(3));
  REQUIRE(GET_REAL(x->grad) == ZERO_R1);
}

TEST_CASE("test_scalar_clamp_complex_grad") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = Tensor::clamp(x, R(1), R(3));
  TensorPtr z = std::make_shared<ComplexScalar>(R(2), true, TEST_DTAG);
  TensorPtr w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(2));

  x = std::make_shared<RealScalar>(ZERO_R1, true, TEST_DTAG);
  y = Tensor::clamp(x, R(1), R(3));
  z = std::make_shared<ComplexScalar>(R(2), true, TEST_DTAG);
  w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == R(1));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), ZERO_R1);

  x = std::make_shared<RealScalar>(R(4), true, TEST_DTAG);
  y = Tensor::clamp(x, R(1), R(3));
  z = std::make_shared<ComplexScalar>(R(2), true, TEST_DTAG);
  w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == R(3));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), ZERO_R1);
}

TEST_CASE("test_real_scalar_abs") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = Tensor::abs(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE(GET_REAL(x->grad) == R(1));

  x = std::make_shared<RealScalar>(R(-2), true, TEST_DTAG);
  y = Tensor::abs(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE(GET_REAL(x->grad) == R(-1));
}

TEST_CASE("test_real_scalar_abs_grad_complex") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = Tensor::abs(x);
  TensorPtr z = std::make_shared<ComplexScalar>(R(2), true, TEST_DTAG);
  TensorPtr w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(2));

  x = std::make_shared<RealScalar>(R(-2), true, TEST_DTAG);
  y = Tensor::abs(x);
  z = std::make_shared<ComplexScalar>(R(2), true, TEST_DTAG);
  w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(-2));
}

TEST_CASE("test_complex_scalar_abs") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(R(2)), true, TEST_DTAG);
  TensorPtr y = Tensor::abs(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1));

  x = std::make_shared<ComplexScalar>(complex(R(-2)), true, TEST_DTAG);
  y = Tensor::abs(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(-1));
}

TEST_CASE("test_complex_scalar_abs_grad_complex") {
  TensorPtr x = std::make_shared<ComplexScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = Tensor::abs(x);
  TensorPtr z = std::make_shared<ComplexScalar>(R(2), true, TEST_DTAG);
  TensorPtr w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(2));

  x = std::make_shared<ComplexScalar>(R(-2), true, TEST_DTAG);
  y = Tensor::abs(x);
  z = std::make_shared<ComplexScalar>(R(2), true, TEST_DTAG);
  w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(-2));
}

TEST_CASE("test_real_scalar_pow") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = x ^ R(3);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(8));
  REQUIRE(GET_REAL(x->grad) == R(12));
}

TEST_CASE("test_complex_scalar_pow") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(R(2)), true, TEST_DTAG);
  TensorPtr y = x ^ R(3);
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), R(8));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(12));
}

TEST_CASE("test_real_scalar_exp") {
  TensorPtr x = std::make_shared<RealScalar>(R(3), true, TEST_DTAG);
  TensorPtr y = R(2) ^ x;
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(8));
  REQUIRE_FLOAT(GET_REAL(x->grad), R(8 * std::log(2)));
}

TEST_CASE("test_complex_scalar_exp") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(R(3)), true, TEST_DTAG);
  TensorPtr y = R(2) ^ x;
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), complex(R(8)));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), complex(R(8 * std::log(2))));
}

TEST_CASE("test_real_scalar_log") {
  TensorPtr x = std::make_shared<RealScalar>(R(8), true, TEST_DTAG);
  TensorPtr y = Tensor::log(x, R(2));
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(3));
  REQUIRE_FLOAT(GET_REAL(x->grad), R(1 / (3 * std::log(2))));
}

TEST_CASE("test_complex_scalar_log") {
  TensorPtr x =
      std::make_shared<ComplexScalar>(complex((real1)8.0f), true, TEST_DTAG);
  TensorPtr y = Tensor::log(x, R(2));
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), complex(R(3)));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), complex(R(1 / (3 * std::log(2)))));
}

TEST_CASE("test_real_scalar_add") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = std::make_shared<RealScalar>(R(3), true, TEST_DTAG);
  TensorPtr z = x + y;
  Tensor::backward(z);

  REQUIRE(GET_REAL(z) == R(5));
  REQUIRE(GET_REAL(x->grad) == R(1));
  REQUIRE(GET_REAL(y->grad) == R(1));
}

TEST_CASE("test_complex_scalar_add") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(R(2)), true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(R(3)), true, TEST_DTAG);
  TensorPtr z = x + y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), R(5));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), R(1));
}

TEST_CASE("test_mixed_scalar_add") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(R(3)), true, TEST_DTAG);
  TensorPtr z = x + y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), R(5));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), R(1));
}

TEST_CASE("test_real_scalar_add_chain") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = std::make_shared<RealScalar>(R(3), true, TEST_DTAG);
  TensorPtr z = std::make_shared<RealScalar>(R(4), true, TEST_DTAG);
  TensorPtr w = x + y;
  TensorPtr i = w + z;
  Tensor::backward(i);

  REQUIRE(GET_REAL(i) == R(9));
  REQUIRE(GET_REAL(x->grad) == R(1));
  REQUIRE(GET_REAL(y->grad) == R(1));
}

TEST_CASE("test_real_scalar_sub") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = std::make_shared<RealScalar>(R(3), true, TEST_DTAG);
  TensorPtr z = x - y;
  Tensor::backward(z);

  REQUIRE(GET_REAL(z) == R(-1));
  REQUIRE(GET_REAL(x->grad) == R(1));
  REQUIRE(GET_REAL(y->grad) == R(-1));
}

TEST_CASE("test_complex_scalar_sub") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(R(2)), true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(R(3)), true, TEST_DTAG);
  TensorPtr z = x - y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), R(-1));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), R(-1));
}

TEST_CASE("test_mixed_scalar_sub") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(R(3)), true, TEST_DTAG);
  TensorPtr z = x - y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), R(-1));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), R(-1));

  x = std::make_shared<ComplexScalar>(complex(R(2)), true, TEST_DTAG);
  y = std::make_shared<RealScalar>(R(3), true, TEST_DTAG);
  z = x - y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), R(-1));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), R(-1));
}

TEST_CASE("test_real_scalar_mul") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = std::make_shared<RealScalar>(R(3), true, TEST_DTAG);
  TensorPtr z = x * y;
  Tensor::backward(z);

  REQUIRE(GET_REAL(z) == R(6));
  REQUIRE(GET_REAL(x->grad) == R(3));
  REQUIRE(GET_REAL(y->grad) == R(2));
}

TEST_CASE("test_complex_scalar_mul") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(R(2)), true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(R(3)), true, TEST_DTAG);
  TensorPtr z = x * y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), complex(R(6)));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), complex(R(3)));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), complex(R(2)));
}

TEST_CASE("test_mixed_scalar_mul") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(R(3)), true, TEST_DTAG);
  TensorPtr z = x * y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), complex(R(6)));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), complex(R(3)));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), complex(R(2)));
}

TEST_CASE("test_real_scalar_div") {
  TensorPtr x = std::make_shared<RealScalar>(R(4), true, TEST_DTAG);
  TensorPtr y = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr z = x / y;
  Tensor::backward(z);

  REQUIRE(GET_REAL(z) == R(2));
  REQUIRE(GET_REAL(x->grad) == R(0.5f));
  REQUIRE(GET_REAL(y->grad) == R(-1));
}

TEST_CASE("test_complex_scalar_div") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(R(4)), true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(R(2)), true, TEST_DTAG);
  TensorPtr z = x / y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(0.5f));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), R(-1));
}

TEST_CASE("test_mixed_scalar_div") {
  TensorPtr x = std::make_shared<RealScalar>(R(4), true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(R(2)), true, TEST_DTAG);
  TensorPtr z = x / y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(0.5f));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), R(-1));

  x = std::make_shared<ComplexScalar>(complex(R(4)), true, TEST_DTAG);
  y = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  z = x / y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(0.5f));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), R(-1));
}

TEST_CASE("test_real_broadcast_mul") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<real1>{R(3), R(4)}, std::vector<vecCapInt>{2},
      std::vector<vecCapInt>{1}, true, TEST_DTAG);
  TensorPtr z = x * y;
  Tensor::backward(z);

  REQUIRE(GET_REAL((*(z.get()))[0]) == R(6));
  REQUIRE(GET_REAL((*(z.get()))[1]) == R(8));
  REQUIRE(GET_REAL(x->grad) == R(7));
}

TEST_CASE("test_complex_broadcast_mul") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(R(2)), true, TEST_DTAG);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<complex>{R(3), R(4)}, std::vector<vecCapInt>{2},
      std::vector<vecCapInt>{1}, true, TEST_DTAG);
  TensorPtr z = x * y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), R(6));
  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[1]), R(8));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(7));
}

TEST_CASE("test_mixed_broadcast_mul") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(R(2)), true, TEST_DTAG);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<real1>{R(3), R(4)}, std::vector<vecCapInt>{2},
      std::vector<vecCapInt>{1}, true, TEST_DTAG);
  TensorPtr z = x * y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), R(6));
  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[1]), R(8));
  REQUIRE_CMPLX(GET_REAL(x->grad), complex(R(7)));
}

TEST_CASE("test_real_matmul") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{R(2), R(3)}, std::vector<vecCapInt>{1, 2},
      std::vector<vecCapInt>{1, 1}, false, TEST_DTAG);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<real1>{R(4), R(5)}, std::vector<vecCapInt>{2, 1},
      std::vector<vecCapInt>{1, 2}, false, TEST_DTAG);
  TensorPtr z = x >> y;

  REQUIRE(GET_REAL((*(z.get()))[0]) == R(23));

  x = std::make_shared<Tensor>(std::vector<real1>{R(2), R(3)},
                               std::vector<vecCapInt>{2, 1},
                               std::vector<vecCapInt>{1, 2}, false, TEST_DTAG);
  y = std::make_shared<Tensor>(std::vector<real1>{R(4), R(5)},
                               std::vector<vecCapInt>{1, 2},
                               std::vector<vecCapInt>{1, 1}, false, TEST_DTAG);
  z = x >> y;

  REQUIRE(GET_REAL((*(*(z.get()))[0])[0]) == R(8));
  REQUIRE(GET_REAL((*(*(z.get()))[0])[1]) == R(12));
  REQUIRE(GET_REAL((*(*(z.get()))[1])[0]) == R(10));
  REQUIRE(GET_REAL((*(*(z.get()))[1])[1]) == R(15));
}

TEST_CASE("test_complex_matmul") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<complex>{R(2), R(3)}, std::vector<vecCapInt>{1, 2},
      std::vector<vecCapInt>{1, 1}, false, TEST_DTAG);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<complex>{R(4), R(5)}, std::vector<vecCapInt>{2, 1},
      std::vector<vecCapInt>{1, 2}, false, TEST_DTAG);
  TensorPtr z = x >> y;

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), R(23));

  x = std::make_shared<Tensor>(std::vector<complex>{R(2), R(3)},
                               std::vector<vecCapInt>{2, 1},
                               std::vector<vecCapInt>{1, 2}, false, TEST_DTAG);
  y = std::make_shared<Tensor>(std::vector<complex>{R(4), R(5)},
                               std::vector<vecCapInt>{1, 2},
                               std::vector<vecCapInt>{1, 1}, false, TEST_DTAG);
  z = x >> y;

  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[0])[0]), R(8));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[0])[1]), R(12));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[1])[0]), R(10));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[1])[1]), R(15));
}

TEST_CASE("test_mixed_matmul") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{R(2), R(3)}, std::vector<vecCapInt>{1, 2},
      std::vector<vecCapInt>{1, 1}, false, TEST_DTAG);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<complex>{R(4), R(5)}, std::vector<vecCapInt>{2, 1},
      std::vector<vecCapInt>{1, 2}, false, TEST_DTAG);
  TensorPtr z = x >> y;

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), R(23));

  x = std::make_shared<Tensor>(std::vector<complex>{R(2), R(3)},
                               std::vector<vecCapInt>{1, 2},
                               std::vector<vecCapInt>{1, 1}, false, TEST_DTAG);
  y = std::make_shared<Tensor>(std::vector<real1>{R(4), R(5)},
                               std::vector<vecCapInt>{2, 1},
                               std::vector<vecCapInt>{1, 2}, false, TEST_DTAG);
  z = y << x;

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), R(23));

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), R(23));

  x = std::make_shared<Tensor>(std::vector<real1>{R(2), R(3)},
                               std::vector<vecCapInt>{2, 1},
                               std::vector<vecCapInt>{1, 2}, false, TEST_DTAG);
  y = std::make_shared<Tensor>(std::vector<complex>{R(4), R(5)},
                               std::vector<vecCapInt>{1, 2},
                               std::vector<vecCapInt>{1, 1}, false, TEST_DTAG);
  z = y << x;

  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[0])[0]), R(8));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[0])[1]), R(12));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[1])[0]), R(10));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[1])[1]), R(15));

  x = std::make_shared<Tensor>(std::vector<complex>{R(2), R(3)},
                               std::vector<vecCapInt>{2, 1},
                               std::vector<vecCapInt>{1, 2}, false, TEST_DTAG);
  y = std::make_shared<Tensor>(std::vector<real1>{R(4), R(5)},
                               std::vector<vecCapInt>{1, 2},
                               std::vector<vecCapInt>{1, 1}, false, TEST_DTAG);
  z = x >> y;

  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[0])[0]), R(8));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[0])[1]), R(12));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[1])[0]), R(10));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[1])[1]), R(15));
}
