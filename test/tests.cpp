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

#include "real_scalar.hpp"

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

TEST_CASE("test_complex") {
  bool test;
  complex cmplx1(ONE_R1, -ONE_R1);
  complex cmplx2((real1)(-0.5f), (real1)0.5f);
  complex cmplx3(ZERO_R1, ZERO_R1);

  REQUIRE(cmplx1 != cmplx2);

  REQUIRE(conj(cmplx1) == complex(ONE_R1, ONE_R1));

  test = ((real1)abs(cmplx1) > (real1)(sqrt(2.0) - EPSILON)) &&
         ((real1)abs(cmplx1) < (real1)(sqrt(2.0) + EPSILON));
  REQUIRE(test);

  cmplx3 = complex(std::polar(ONE_R1, PI_R1 / (real1)2.0f));
  test = (real(cmplx3) > (real1)(0.0 - EPSILON)) &&
         (real(cmplx3) < (real1)(0.0 + EPSILON));
  REQUIRE(test);
  test = (imag(cmplx3) > (real1)(1.0 - EPSILON)) &&
         (imag(cmplx3) < (real1)(1.0 + EPSILON));
  REQUIRE(test);

  cmplx3 = cmplx1 + cmplx2;
  test = (real(cmplx3) > (real1)(0.5 - EPSILON)) &&
         (real(cmplx3) < (real1)(0.5 + EPSILON));
  REQUIRE(test);
  test = (imag(cmplx3) > (real1)(-0.5 - EPSILON)) &&
         (imag(cmplx3) < (real1)(-0.5 + EPSILON));
  REQUIRE(test);

  cmplx3 = cmplx1 - cmplx2;
  test = (real(cmplx3) > (real1)(1.5 - EPSILON)) &&
         (real(cmplx3) < (real1)(1.5 + EPSILON));
  REQUIRE(test);
  test = (imag(cmplx3) > (real1)(-1.5 - EPSILON)) &&
         (imag(cmplx3) < (real1)(-1.5 + EPSILON));
  REQUIRE(test);

  cmplx3 = cmplx1 * cmplx2;
  test = (real(cmplx3) > (real1)(0.0 - EPSILON)) &&
         (real(cmplx3) < (real1)(0.0 + EPSILON));
  REQUIRE(test);
  test = (imag(cmplx3) > (real1)(1.0 - EPSILON)) &&
         (imag(cmplx3) < (real1)(1.0 + EPSILON));
  REQUIRE(test);

  cmplx3 = cmplx1;
  cmplx3 *= cmplx2;
  test = (real(cmplx3) > (real1)(0.0 - EPSILON)) &&
         (real(cmplx3) < (real1)(0.0 + EPSILON));
  REQUIRE(test);
  test = (imag(cmplx3) > (real1)(1.0 - EPSILON)) &&
         (imag(cmplx3) < (real1)(1.0 + EPSILON));
  REQUIRE(test);

  cmplx3 = cmplx1 / cmplx2;
  test = (real(cmplx3) > (real1)(-2.0 - EPSILON)) &&
         (real(cmplx3) < (real1)(-2.0 + EPSILON));
  REQUIRE(test);
  test = (imag(cmplx3) > (real1)(0.0 - EPSILON)) &&
         (imag(cmplx3) < (real1)(0.0 + EPSILON));
  REQUIRE(test);

  cmplx3 = cmplx2;
  cmplx3 /= cmplx1;
  test = (real(cmplx3) > (real1)(-0.5 - EPSILON)) &&
         (real(cmplx3) < (real1)(-0.5 + EPSILON));
  REQUIRE(test);
  test = (imag(cmplx3) > (real1)(0.0 - EPSILON)) &&
         (imag(cmplx3) < (real1)(0.0 + EPSILON));
  REQUIRE(test);

  cmplx3 = ((real1)2.0) * cmplx1;
  test = (real(cmplx3) > (real1)(2.0 - EPSILON)) &&
         (real(cmplx3) < (real1)(2.0 + EPSILON));
  REQUIRE(test);
  test = (imag(cmplx3) > (real1)(-2.0 - EPSILON)) &&
         (imag(cmplx3) < (real1)(-2.0 + EPSILON));
  REQUIRE(test);
}

TEST_CASE("test_scalar_grad") {
  TensorPtr x = std::make_shared<RealScalar>(2.0, true, TEST_DTAG);
  TensorPtr y = std::make_shared<RealScalar>(3.0, true, TEST_DTAG);
  TensorPtr z = x * y;
  Tensor::backward(z);

  REQUIRE(GET_REAL(x->grad) == (ONE_R1 * 3));
  REQUIRE(GET_REAL(y->grad) == (ONE_R1 * 2));
}
