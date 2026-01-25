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
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#include <iostream>

#include "catch.hpp"

#include "tests.hpp"

#include "autograd/sgd.hpp"
#include "autograd/zero_grad.hpp"
#include "tensors/complex_scalar.hpp"
#include "tensors/real_scalar.hpp"

using namespace Weed;

#define EPSILON 0.01f
#define REQUIRE_FLOAT(A, B)                                                    \
  do {                                                                         \
    real1_f __tmp_a = A;                                                       \
    real1_f __tmp_b = B;                                                       \
    REQUIRE(__tmp_a < (__tmp_b + EPSILON));                                    \
    REQUIRE(__tmp_a > (__tmp_b - EPSILON));                                    \
  } while (0);
#define REQUIRE_CMPLX(A, B)                                                    \
  do {                                                                         \
    complex __tmp_a = A;                                                       \
    complex __tmp_b = B;                                                       \
    REQUIRE(std::norm(__tmp_a - __tmp_b) < EPSILON);                           \
  } while (0);

#define GET_REAL(ptr) static_cast<RealScalar *>((ptr).get())->get_item()
#define GET_COMPLEX(ptr) static_cast<ComplexScalar *>((ptr).get())->get_item()

TEST_CASE("test_sum_real") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{1.0, 2.0, 3.0}, std::vector<vecCapInt>{3, 1},
      std::vector<vecCapInt>{1, 3}, true, TEST_DTAG);
  TensorPtr y = Tensor::sum(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == (ONE_R1 * 6));
  REQUIRE(GET_REAL(x->grad) == ONE_R1);
}

TEST_CASE("test_sum_complex") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<complex>{1.0, 2.0, 3.0}, std::vector<vecCapInt>{3, 1},
      std::vector<vecCapInt>{1, 3}, true, TEST_DTAG);
  TensorPtr y = Tensor::sum(x);
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), (ONE_R1 * 6));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), ONE_R1);
}

TEST_CASE("test_mean_real") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{1.0, 2.0, 3.0}, std::vector<vecCapInt>{3, 1},
      std::vector<vecCapInt>{1, 3}, true, TEST_DTAG);
  TensorPtr y = Tensor::mean(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == (ONE_R1 * 2));
  REQUIRE(GET_REAL(x->grad) == (ONE_R1 / 3));
}

TEST_CASE("test_mean_complex") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<complex>{1.0, 2.0, 3.0}, std::vector<vecCapInt>{3, 1},
      std::vector<vecCapInt>{1, 3}, true, TEST_DTAG);
  TensorPtr y = Tensor::mean(x);
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), (ONE_R1 * 2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), (ONE_R1 / 3));
}

TEST_CASE("test_scalar_relu") {
  TensorPtr x = std::make_shared<RealScalar>(2.0, true, TEST_DTAG);
  TensorPtr y = Tensor::relu(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == (ONE_R1 * 2));
  REQUIRE(GET_REAL(x->grad) == ONE_R1);

  x = std::make_shared<RealScalar>(-2.0, true, TEST_DTAG);
  y = Tensor::relu(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == ZERO_CMPLX);
  REQUIRE(GET_REAL(x->grad) == ZERO_CMPLX);
}

TEST_CASE("test_real_scalar_abs") {
  TensorPtr x = std::make_shared<RealScalar>(2.0, true, TEST_DTAG);
  TensorPtr y = Tensor::abs(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == (ONE_R1 * 2));
  REQUIRE(GET_REAL(x->grad) == ONE_R1);

  x = std::make_shared<RealScalar>(-2.0, true, TEST_DTAG);
  y = Tensor::abs(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == (ONE_R1 * 2));
  REQUIRE(GET_REAL(x->grad) == -ONE_R1);
}

TEST_CASE("test_complex_scalar_abs") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(2.0), true, TEST_DTAG);
  TensorPtr y = Tensor::abs(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == (ONE_R1 * 2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), ONE_R1);

  x = std::make_shared<ComplexScalar>(complex(-2.0), true, TEST_DTAG);
  y = Tensor::abs(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == (ONE_R1 * 2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), -ONE_R1);
}

TEST_CASE("test_real_scalar_pow") {
  TensorPtr x = std::make_shared<RealScalar>(2.0, true, TEST_DTAG);
  TensorPtr y = x ^ 3.0;
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == (ONE_R1 * 8));
  REQUIRE(GET_REAL(x->grad) == (ONE_R1 * 12));
}

TEST_CASE("test_complex_scalar_pow") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(2.0), true, TEST_DTAG);
  TensorPtr y = x ^ 3.0;
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), (ONE_R1 * 8));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), (ONE_R1 * 12));
}

TEST_CASE("test_real_scalar_exp") {
  TensorPtr x = std::make_shared<RealScalar>(3.0, true, TEST_DTAG);
  TensorPtr y = 2.0 ^ x;
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == (ONE_R1 * 8));
  REQUIRE_FLOAT(GET_REAL(x->grad), (ONE_R1 * 8 * std::log(2)));
}

TEST_CASE("test_complex_scalar_exp") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(3.0), true, TEST_DTAG);
  TensorPtr y = 2.0 ^ x;
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), (ONE_R1 * 8));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), (ONE_R1 * 8 * std::log(2)));
}

TEST_CASE("test_real_scalar_log") {
  TensorPtr x = std::make_shared<RealScalar>(8.0, true, TEST_DTAG);
  TensorPtr y = Tensor::log(x, 2.0);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == (ONE_R1 * 3));
  REQUIRE_FLOAT(GET_REAL(x->grad), (ONE_R1 / (3 * std::log(2))));
}

TEST_CASE("test_complex_scalar_log") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(8.0), true, TEST_DTAG);
  TensorPtr y = Tensor::log(x, 2.0);
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), (ONE_R1 * 3));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), (ONE_R1 / (3 * std::log(2))));
}

TEST_CASE("test_real_scalar_add") {
  TensorPtr x = std::make_shared<RealScalar>(2.0, true, TEST_DTAG);
  TensorPtr y = std::make_shared<RealScalar>(3.0, true, TEST_DTAG);
  TensorPtr z = x + y;
  Tensor::backward(z);

  REQUIRE(GET_REAL(z) == (ONE_R1 * 5));
  REQUIRE(GET_REAL(x->grad) == ONE_R1);
  REQUIRE(GET_REAL(y->grad) == ONE_R1);
}

TEST_CASE("test_complex_scalar_add") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(2.0), true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(3.0), true, TEST_DTAG);
  TensorPtr z = x + y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), (ONE_R1 * 5));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), ONE_R1);
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), ONE_R1);
}

TEST_CASE("test_mixed_scalar_add") {
  TensorPtr x = std::make_shared<RealScalar>(2.0, true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(3.0), true, TEST_DTAG);
  TensorPtr z = x + y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), (ONE_R1 * 5));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), ONE_R1);
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), ONE_R1);
}

TEST_CASE("test_real_scalar_sub") {
  TensorPtr x = std::make_shared<RealScalar>(2.0, true, TEST_DTAG);
  TensorPtr y = std::make_shared<RealScalar>(3.0, true, TEST_DTAG);
  TensorPtr z = x - y;
  Tensor::backward(z);

  REQUIRE(GET_REAL(z) == -ONE_R1);
  REQUIRE(GET_REAL(x->grad) == ONE_R1);
  REQUIRE(GET_REAL(y->grad) == -ONE_R1);
}

TEST_CASE("test_complex_scalar_sub") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(2.0), true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(3.0), true, TEST_DTAG);
  TensorPtr z = x - y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), -ONE_R1);
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), ONE_R1);
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), -ONE_R1);
}

TEST_CASE("test_mixed_scalar_sub") {
  TensorPtr x = std::make_shared<RealScalar>(2.0, true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(3.0), true, TEST_DTAG);
  TensorPtr z = x - y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), -ONE_R1);
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), ONE_R1);
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), -ONE_R1);

  x = std::make_shared<ComplexScalar>(complex(2.0), true, TEST_DTAG);
  y = std::make_shared<RealScalar>(3.0, true, TEST_DTAG);
  z = x - y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), -ONE_R1);
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), ONE_R1);
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), -ONE_R1);
}

TEST_CASE("test_real_scalar_mul") {
  TensorPtr x = std::make_shared<RealScalar>(2.0, true, TEST_DTAG);
  TensorPtr y = std::make_shared<RealScalar>(3.0, true, TEST_DTAG);
  TensorPtr z = x * y;
  Tensor::backward(z);

  REQUIRE(GET_REAL(z) == (ONE_R1 * 6));
  REQUIRE(GET_REAL(x->grad) == (ONE_R1 * 3));
  REQUIRE(GET_REAL(y->grad) == (ONE_R1 * 2));
}

TEST_CASE("test_complex_scalar_mul") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(2.0), true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(3.0), true, TEST_DTAG);
  TensorPtr z = x * y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), (ONE_R1 * 6));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), (ONE_R1 * 3));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), (ONE_R1 * 2));
}

TEST_CASE("test_mixed_scalar_mul") {
  TensorPtr x = std::make_shared<RealScalar>(2.0, true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(3.0), true, TEST_DTAG);
  TensorPtr z = x * y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), (ONE_R1 * 6));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), (ONE_R1 * 3));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), (ONE_R1 * 2));
}

TEST_CASE("test_real_scalar_div") {
  TensorPtr x = std::make_shared<RealScalar>(4.0, true, TEST_DTAG);
  TensorPtr y = std::make_shared<RealScalar>(2.0, true, TEST_DTAG);
  TensorPtr z = x / y;
  Tensor::backward(z);

  REQUIRE(GET_REAL(z) == (ONE_R1 * 2));
  REQUIRE(GET_REAL(x->grad) == (ONE_R1 / 2));
  REQUIRE(GET_REAL(y->grad) == -ONE_R1);
}

TEST_CASE("test_complex_scalar_div") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(4.0), true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(2.0), true, TEST_DTAG);
  TensorPtr z = x / y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), (ONE_R1 * 2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), (ONE_R1 / 2));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), -ONE_R1);
}

TEST_CASE("test_mixed_scalar_div") {
  TensorPtr x = std::make_shared<RealScalar>(4.0, true, TEST_DTAG);
  TensorPtr y = std::make_shared<ComplexScalar>(complex(2.0), true, TEST_DTAG);
  TensorPtr z = x / y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), (ONE_R1 * 2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), (ONE_R1 / 2));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), -ONE_R1);

  x = std::make_shared<ComplexScalar>(complex(4.0), true, TEST_DTAG);
  y = std::make_shared<RealScalar>(2.0, true, TEST_DTAG);
  z = x / y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX(z), (ONE_R1 * 2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), (ONE_R1 / 2));
  REQUIRE_CMPLX(GET_COMPLEX(y->grad), -ONE_R1);
}

TEST_CASE("test_real_broadcast_mul") {
  TensorPtr x = std::make_shared<RealScalar>(2.0, false, TEST_DTAG);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<real1>{3.0, 4.0}, std::vector<vecCapInt>{2},
      std::vector<vecCapInt>{1}, false, TEST_DTAG);
  TensorPtr z = x * y;

  REQUIRE(GET_REAL((*(z.get()))[0]) == (ONE_R1 * 6));
  REQUIRE(GET_REAL((*(z.get()))[1]) == (ONE_R1 * 8));
}

TEST_CASE("test_complex_broadcast_mul") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(2.0), false, TEST_DTAG);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<complex>{3.0, 4.0}, std::vector<vecCapInt>{2},
      std::vector<vecCapInt>{1}, false, TEST_DTAG);
  TensorPtr z = x * y;

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), (ONE_R1 * 6));
  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[1]), (ONE_R1 * 8));
}

TEST_CASE("test_mixed_broadcast_mul") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(2.0), false, TEST_DTAG);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<real1>{3.0, 4.0}, std::vector<vecCapInt>{2},
      std::vector<vecCapInt>{1}, false, TEST_DTAG);
  TensorPtr z = x * y;

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), (ONE_R1 * 6));
  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[1]), (ONE_R1 * 8));
}

TEST_CASE("test_real_matmul") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{2.0, 3.0}, std::vector<vecCapInt>{1, 2},
      std::vector<vecCapInt>{1, 1}, false, TEST_DTAG);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<real1>{4.0, 5.0}, std::vector<vecCapInt>{2, 1},
      std::vector<vecCapInt>{1, 2}, false, TEST_DTAG);
  TensorPtr z = x >> y;

  REQUIRE(GET_REAL((*(z.get()))[0]) == (ONE_R1 * 23));
}

TEST_CASE("test_complex_matmul") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<complex>{2.0, 3.0}, std::vector<vecCapInt>{1, 2},
      std::vector<vecCapInt>{1, 1}, false, TEST_DTAG);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<complex>{4.0, 5.0}, std::vector<vecCapInt>{2, 1},
      std::vector<vecCapInt>{1, 2}, false, TEST_DTAG);
  TensorPtr z = x >> y;

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), (ONE_R1 * 23));
}

TEST_CASE("test_mixed_matmul") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{2.0, 3.0}, std::vector<vecCapInt>{1, 2},
      std::vector<vecCapInt>{1, 1}, false, TEST_DTAG);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<complex>{4.0, 5.0}, std::vector<vecCapInt>{2, 1},
      std::vector<vecCapInt>{1, 2}, false, TEST_DTAG);
  TensorPtr z = x >> y;

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), (ONE_R1 * 23));

  x = std::make_shared<Tensor>(std::vector<complex>{2.0, 3.0},
                               std::vector<vecCapInt>{1, 2},
                               std::vector<vecCapInt>{1, 1}, false, TEST_DTAG);
  y = std::make_shared<Tensor>(std::vector<real1>{4.0, 5.0},
                               std::vector<vecCapInt>{2, 1},
                               std::vector<vecCapInt>{1, 2}, false, TEST_DTAG);
  z = x >> y;

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), (ONE_R1 * 23));
}
