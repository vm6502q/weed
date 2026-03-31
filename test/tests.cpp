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
#include "storage/all_storage.hpp"
#include "tensors/complex_scalar.hpp"
#include "tensors/real_scalar.hpp"

using namespace Weed;

#define R(v) real1(v)
#define C(v) complex(R(v))

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

TEST_CASE("test_fill_value_real") {
  TensorPtr x = std::make_shared<RealScalar>(R(1), true, DeviceTag::CPU);
  static_cast<CpuRealStorage *>(x->storage.get())->FillValue(R(2));
  REQUIRE(GET_REAL(x) == R(2));

  x = std::make_shared<ComplexScalar>(C(1), true, DeviceTag::CPU);
  static_cast<CpuComplexStorage *>(x->storage.get())->FillValue(R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x), R(2));

#if ENABLE_GPU
  x = std::make_shared<RealScalar>(R(1), true, DeviceTag::GPU);
  static_cast<GpuRealStorage *>(x->storage.get())->FillValue(R(2));
  REQUIRE(GET_REAL(x) == R(2));

  x = std::make_shared<ComplexScalar>(C(1), true, DeviceTag::GPU);
  static_cast<GpuComplexStorage *>(x->storage.get())->FillValue(R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x), R(2));
#endif
}

#if ENABLE_GPU
TEST_CASE("test_real_cpu_gpu_conversions") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), false, DeviceTag::CPU);
  x->storage = x->storage->gpu();

  REQUIRE(GET_REAL(x) == R(2));

  x = std::make_shared<RealScalar>(R(2), false, DeviceTag::GPU);
  x->storage = x->storage->cpu();

  REQUIRE(GET_REAL(x) == R(2));
}

TEST_CASE("test_complex_cpu_gpu_conversions") {
  TensorPtr x = std::make_shared<ComplexScalar>(C(2), false, DeviceTag::CPU);
  x->storage = x->storage->gpu();

  REQUIRE_CMPLX(GET_COMPLEX(x), R(2));

  x = std::make_shared<ComplexScalar>(C(2), false, DeviceTag::GPU);
  x->storage = x->storage->cpu();

  REQUIRE_CMPLX(GET_COMPLEX(x), R(2));
}
#endif

TEST_CASE("test_sum_real") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::sum(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(6));
  REQUIRE(GET_REAL(x->grad) == R(1));
}

TEST_CASE("test_sum_complex") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<complex>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::sum(x);
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), R(6));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1));
}

TEST_CASE("test_mean_real") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::mean(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE(GET_REAL(x->grad) == R(R(1) / R(3)));
}

TEST_CASE("test_mean_complex") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<complex>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::mean(x);
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(R(1) / R(3)));
}

TEST_CASE("test_variance_real") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(1), R(2)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::variance(x);
  Tensor::backward(y);

  REQUIRE_FLOAT(GET_REAL(y), R(0.03704));
  REQUIRE_FLOAT(GET_REAL(x->grad), R(R(-2) / R(9)));
}

TEST_CASE("test_variance_complex") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<complex>{R(1), R(1), R(2)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::variance(x);
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), R(0.03704));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(R(-2) / R(9)));
}

TEST_CASE("test_stddev_real") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(1), R(2)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::stddev(x);
  Tensor::backward(y);

  REQUIRE_FLOAT(GET_REAL(y), R(sqrt(0.03704)));
  REQUIRE_FLOAT(GET_REAL(x->grad), R(-0.3849));
}

TEST_CASE("test_stddev_complex") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<complex>{R(1), R(1), R(2)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::stddev(x);
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), R(sqrt(0.03704)));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(-0.3849));
}

TEST_CASE("test_sum_axis_real") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{R(1), R(1)}, std::vector<tcapint>{2}, true, TEST_DTAG);
  TensorPtr y = Tensor::sum(x, 0U);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(2));

  RealStorage *xg = static_cast<RealStorage *>(x->grad->storage.get());
  REQUIRE((*xg)[0] == R(1));
  REQUIRE((*xg)[1] == R(1));
}

TEST_CASE("test_sum_axis_complex") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<complex>{R(1), R(1)},
                               std::vector<tcapint>{2}, true, TEST_DTAG);
  TensorPtr y = Tensor::sum(x, 0U);
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), R(2));

  ComplexStorage *xg = static_cast<ComplexStorage *>(x->grad->storage.get());
  REQUIRE((*xg)[0] == R(1));
  REQUIRE((*xg)[1] == R(1));
}

TEST_CASE("test_max_axis") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{R(0), R(1)}, std::vector<tcapint>{2}, true, TEST_DTAG);
  TensorPtr y = Tensor::max(x, 0U);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(1));

  RealStorage *xg = static_cast<RealStorage *>(x->grad->storage.get());
  REQUIRE((*xg)[0] == R(0));
  REQUIRE((*xg)[1] == R(1));
}

TEST_CASE("test_min_axis") {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{R(0), R(1)}, std::vector<tcapint>{2}, true, TEST_DTAG);
  TensorPtr y = Tensor::min(x, 0U);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(0));

  RealStorage *xg = static_cast<RealStorage *>(x->grad->storage.get());
  REQUIRE((*xg)[0] == R(1));
  REQUIRE((*xg)[1] == R(0));
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

TEST_CASE("test_scalar_relu_mixed_grad") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = Tensor::relu(x);
  x->grad->upcast(DType::COMPLEX);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1));

  x = std::make_shared<RealScalar>(R(-2), true, TEST_DTAG);
  y = Tensor::relu(x);
  x->grad->upcast(DType::COMPLEX);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == ZERO_CMPLX);
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), ZERO_CMPLX);
}

TEST_CASE("test_scalar_sigmoid") {
  TensorPtr x = std::make_shared<RealScalar>(R(0), true, TEST_DTAG);
  TensorPtr y = Tensor::sigmoid(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(0.5));
  REQUIRE(GET_REAL(x->grad) == R(0.25));
}

TEST_CASE("test_scalar_sigmoid_complex_grad") {
  TensorPtr x = std::make_shared<RealScalar>(R(0), true, TEST_DTAG);
  TensorPtr y = Tensor::sigmoid(x);
  TensorPtr z = std::make_shared<ComplexScalar>(R(1), true, TEST_DTAG);
  TensorPtr w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == R(0.5));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(0.25));
}

TEST_CASE("test_scalar_sigmoid_mixed_grad") {
  TensorPtr x = std::make_shared<RealScalar>(R(0), true, TEST_DTAG);
  TensorPtr y = Tensor::sigmoid(x);
  x->grad->upcast(DType::COMPLEX);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(0.5));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(0.25));
}

TEST_CASE("test_scalar_tanh") {
  TensorPtr x = std::make_shared<RealScalar>(R(0), true, TEST_DTAG);
  TensorPtr y = Tensor::tanh(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(0));
  REQUIRE(GET_REAL(x->grad) == R(1.0));
}

TEST_CASE("test_scalar_tanh_complex_grad") {
  TensorPtr x = std::make_shared<RealScalar>(R(0), true, TEST_DTAG);
  TensorPtr y = Tensor::tanh(x);
  TensorPtr z = std::make_shared<ComplexScalar>(R(1), true, TEST_DTAG);
  TensorPtr w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == R(0));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1.0));
}

TEST_CASE("test_scalar_tanh_mixed_grad") {
  TensorPtr x = std::make_shared<RealScalar>(R(0), true, TEST_DTAG);
  TensorPtr y = Tensor::tanh(x);
  x->grad->upcast(DType::COMPLEX);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(0));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1.0));
}

TEST_CASE("test_scalar_sin") {
  TensorPtr x = std::make_shared<RealScalar>(R(0), true, TEST_DTAG);
  TensorPtr y = Tensor::sin(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(0));
  REQUIRE(GET_REAL(x->grad) == R(1.0));
}

TEST_CASE("test_scalar_sin_complex_grad") {
  TensorPtr x = std::make_shared<RealScalar>(R(0), true, TEST_DTAG);
  TensorPtr y = Tensor::sin(x);
  TensorPtr z = std::make_shared<ComplexScalar>(R(1), true, TEST_DTAG);
  TensorPtr w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == R(0));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1.0));
}

TEST_CASE("test_scalar_sin_mixed_grad") {
  TensorPtr x = std::make_shared<RealScalar>(R(0), true, TEST_DTAG);
  TensorPtr y = Tensor::sin(x);
  x->grad->upcast(DType::COMPLEX);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(0));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1.0));
}

TEST_CASE("test_scalar_cos") {
  TensorPtr x = std::make_shared<RealScalar>(R(0), true, TEST_DTAG);
  TensorPtr y = Tensor::cos(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(1.0));
  REQUIRE(GET_REAL(x->grad) == R(0));
}

TEST_CASE("test_scalar_cos_complex_grad") {
  TensorPtr x = std::make_shared<RealScalar>(R(0), true, TEST_DTAG);
  TensorPtr y = Tensor::cos(x);
  TensorPtr z = std::make_shared<ComplexScalar>(R(1), true, TEST_DTAG);
  TensorPtr w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == R(1.0));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(0));
}

TEST_CASE("test_scalar_cos_mixed_grad") {
  TensorPtr x = std::make_shared<RealScalar>(R(0), true, TEST_DTAG);
  TensorPtr y = Tensor::cos(x);
  x->grad->upcast(DType::COMPLEX);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(1.0));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(0));
}

TEST_CASE("test_max") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::max(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(3));

  RealStorage *xg = static_cast<RealStorage *>(x->grad->storage.get());
  REQUIRE((*xg)[0] == R(0));
  REQUIRE((*xg)[1] == R(0));
  REQUIRE((*xg)[2] == R(1));
}

TEST_CASE("test_max_complex_grad") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::max(x);
  TensorPtr z = std::make_shared<ComplexScalar>(R(1), true, TEST_DTAG);
  TensorPtr w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == R(3));

  ComplexStorage *xg = static_cast<ComplexStorage *>(x->grad->storage.get());
  REQUIRE_CMPLX((*xg)[0], R(0));
  REQUIRE_CMPLX((*xg)[1], R(0));
  REQUIRE_CMPLX((*xg)[2], R(1));
}

TEST_CASE("test_max_mixed_grad") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::max(x);
  x->grad->upcast(DType::COMPLEX);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(3));

  ComplexStorage *xg = static_cast<ComplexStorage *>(x->grad->storage.get());
  REQUIRE_CMPLX((*xg)[0], R(0));
  REQUIRE_CMPLX((*xg)[1], R(0));
  REQUIRE_CMPLX((*xg)[2], R(1));
}

TEST_CASE("test_min") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::min(x);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(1));

  RealStorage *xg = static_cast<RealStorage *>(x->grad->storage.get());
  REQUIRE((*xg)[0] == R(1));
  REQUIRE((*xg)[1] == R(0));
  REQUIRE((*xg)[2] == R(0));
}

TEST_CASE("test_min_complex_grad") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::min(x);
  TensorPtr z = std::make_shared<ComplexScalar>(R(1), true, TEST_DTAG);
  TensorPtr w = y * z;
  Tensor::backward(w);

  REQUIRE(GET_REAL(y) == R(1));

  ComplexStorage *xg = static_cast<ComplexStorage *>(x->grad->storage.get());
  REQUIRE_CMPLX((*xg)[0], R(1));
  REQUIRE_CMPLX((*xg)[1], R(0));
  REQUIRE_CMPLX((*xg)[2], R(0));
}

TEST_CASE("test_min_mixed_grad") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::min(x);
  x->grad->upcast(DType::COMPLEX);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(1));

  ComplexStorage *xg = static_cast<ComplexStorage *>(x->grad->storage.get());
  REQUIRE_CMPLX((*xg)[0], R(1));
  REQUIRE_CMPLX((*xg)[1], R(0));
  REQUIRE_CMPLX((*xg)[2], R(0));
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

TEST_CASE("test_scalar_clamp_mixed_grad") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = Tensor::clamp(x, R(1), R(3));
  x->grad->upcast(DType::COMPLEX);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1));

  x = std::make_shared<RealScalar>(ZERO_R1, true, TEST_DTAG);
  y = Tensor::clamp(x, R(1), R(3));
  x->grad->upcast(DType::COMPLEX);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(1));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), ZERO_R1);

  x = std::make_shared<RealScalar>(R(4), true, TEST_DTAG);
  y = Tensor::clamp(x, R(1), R(3));
  x->grad->upcast(DType::COMPLEX);
  Tensor::backward(y);

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

  x = std::make_shared<RealScalar>(R(-2), true, TEST_DTAG);
  y = Tensor::abs(x);
  TensorPtr z = std::make_shared<ComplexScalar>(C(2), true, TEST_DTAG);
  TensorPtr w = y + z;
  Tensor::backward(w);

  REQUIRE_CMPLX(GET_COMPLEX(w), C(4));
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

TEST_CASE("test_real_scalar_abs_mixed_grad") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = Tensor::abs(x);
  x->grad->upcast(DType::COMPLEX);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(1));

  x = std::make_shared<RealScalar>(R(-2), true, TEST_DTAG);
  y = Tensor::abs(x);
  x->grad->upcast(DType::COMPLEX);
  Tensor::backward(y);

  REQUIRE(GET_REAL(y) == R(2));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(-1));
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

  REQUIRE_FLOAT(GET_REAL(y), R(8));
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

  REQUIRE_FLOAT(GET_REAL(y), R(3));
  REQUIRE_FLOAT(GET_REAL(x->grad), R(1 / (8 * std::log(2))));
}

TEST_CASE("test_complex_scalar_log") {
  TensorPtr x =
      std::make_shared<ComplexScalar>(complex((real1)8.0f), true, TEST_DTAG);
  TensorPtr y = Tensor::log(x, R(2));
  Tensor::backward(y);

  REQUIRE_CMPLX(GET_COMPLEX(y), complex(R(3)));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), complex(R(1 / (8 * std::log(2)))));
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

TEST_CASE("test_mixed_scalar_add_in_place") {
  TensorPtr x =
      std::make_shared<ComplexScalar>(complex(R(2)), false, TEST_DTAG);
  TensorPtr y = std::make_shared<RealScalar>(R(3), false, TEST_DTAG);
  Weed::add_in_place(*(x.get()), *(y.get()));

  REQUIRE_CMPLX(GET_COMPLEX(x), R(5));
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

TEST_CASE("test_real_scalar_add_sparse") {
  RealSparseVector xvec;
  xvec[2] = R(2);

  ComplexSparseVector yvec;
  yvec[1] = R(1);

  TensorPtr x = std::make_shared<Tensor>(xvec, std::vector<tcapint>{3}, false);
  TensorPtr y = std::make_shared<Tensor>(yvec, std::vector<tcapint>{3}, false);
  TensorPtr z = x + y;

  ComplexStorage *zs = static_cast<ComplexStorage *>(z->storage.get());
  REQUIRE_CMPLX((*zs)[0], R(0));
  REQUIRE_CMPLX((*zs)[1], R(1));
  REQUIRE_CMPLX((*zs)[2], R(2));
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

TEST_CASE("test_mixed_scalar_sub_in_place") {
  TensorPtr x =
      std::make_shared<ComplexScalar>(complex(R(2)), false, TEST_DTAG);
  TensorPtr y = std::make_shared<RealScalar>(R(3), false, TEST_DTAG);
  Weed::sub_in_place(*(x.get()), *(y.get()));

  REQUIRE_CMPLX(GET_COMPLEX(x), R(-1));
}

TEST_CASE("test_real_scalar_sub_sparse") {
  RealSparseVector xvec;
  xvec[1] = R(2);

  RealSparseVector yvec;
  yvec[1] = R(1);

  TensorPtr x = std::make_shared<Tensor>(xvec, std::vector<tcapint>{3}, false);
  TensorPtr y = std::make_shared<Tensor>(yvec, std::vector<tcapint>{3}, false);
  TensorPtr z = x - y;

  RealStorage *zs = static_cast<RealStorage *>(z->storage.get());
  REQUIRE((*zs)[0] == R(0));
  REQUIRE((*zs)[1] == R(1));
  REQUIRE((*zs)[2] == R(0));
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

TEST_CASE("test_real_scalar_mul_sparse") {
  RealSparseVector xvec;
  xvec[1] = R(2);
  xvec[2] = R(1);

  RealSparseVector yvec;
  yvec[1] = R(1);

  TensorPtr x = std::make_shared<Tensor>(xvec, std::vector<tcapint>{3}, false);
  TensorPtr y = std::make_shared<Tensor>(yvec, std::vector<tcapint>{3}, false);
  TensorPtr z = x * y;

  RealStorage *zs = static_cast<RealStorage *>(z->storage.get());
  REQUIRE((*zs)[0] == R(0));
  REQUIRE((*zs)[1] == R(2));
  REQUIRE((*zs)[2] == R(0));
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

TEST_CASE("test_real_scalar_div_sparse") {
  RealSparseVector xvec;
  xvec[1] = R(4);
  RealSparseVector yvec;
  yvec[0] = R(2);
  yvec[1] = R(2);
  yvec[2] = R(2);

  TensorPtr x = std::make_shared<Tensor>(xvec, std::vector<tcapint>{3}, false);
  TensorPtr y = std::make_shared<Tensor>(yvec, std::vector<tcapint>{3}, false);
  TensorPtr z = x / y;

  RealStorage *zs = static_cast<RealStorage *>(z->storage.get());
  REQUIRE((*zs)[0] == R(0));
  REQUIRE((*zs)[1] == R(2));
  REQUIRE((*zs)[2] == R(0));
}

TEST_CASE("test_real_broadcast_mul") {
  TensorPtr x = std::make_shared<RealScalar>(R(2), true, TEST_DTAG);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<real1>{R(3), R(4)}, std::vector<tcapint>{2}, true, TEST_DTAG);
  TensorPtr z = x * y;
  Tensor::backward(z);

  REQUIRE(GET_REAL((*(z.get()))[0]) == R(6));
  REQUIRE(GET_REAL((*(z.get()))[1]) == R(8));
  REQUIRE(GET_REAL(x->grad) == R(7));
}

TEST_CASE("test_complex_broadcast_mul") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(R(2)), true, TEST_DTAG);
  TensorPtr y =
      std::make_shared<Tensor>(std::vector<complex>{R(3), R(4)},
                               std::vector<tcapint>{2}, true, TEST_DTAG);
  TensorPtr z = x * y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), R(6));
  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[1]), R(8));
  REQUIRE_CMPLX(GET_COMPLEX(x->grad), R(7));
}

TEST_CASE("test_mixed_broadcast_mul") {
  TensorPtr x = std::make_shared<ComplexScalar>(complex(R(2)), true, TEST_DTAG);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<real1>{R(3), R(4)}, std::vector<tcapint>{2}, true, TEST_DTAG);
  TensorPtr z = x * y;
  Tensor::backward(z);

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), R(6));
  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[1]), R(8));
  REQUIRE_CMPLX(GET_REAL(x->grad), complex(R(7)));
}

TEST_CASE("test_real_matmul") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(2), R(3)},
                               std::vector<tcapint>{1, 2}, false, TEST_DTAG);
  TensorPtr y =
      std::make_shared<Tensor>(std::vector<real1>{R(4), R(5)},
                               std::vector<tcapint>{2, 1}, false, TEST_DTAG);
  TensorPtr z = x >> y;

  REQUIRE(GET_REAL((*(z.get()))[0]) == R(23));

  x = std::make_shared<Tensor>(std::vector<real1>{R(2), R(3)},
                               std::vector<tcapint>{2, 1}, false, TEST_DTAG);
  y = std::make_shared<Tensor>(std::vector<real1>{R(4), R(5)},
                               std::vector<tcapint>{1, 2}, false, TEST_DTAG);
  z = x >> y;

  REQUIRE(GET_REAL((*(*(z.get()))[0])[0]) == R(8));
  REQUIRE(GET_REAL((*(*(z.get()))[0])[1]) == R(12));
  REQUIRE(GET_REAL((*(*(z.get()))[1])[0]) == R(10));
  REQUIRE(GET_REAL((*(*(z.get()))[1])[1]) == R(15));
}

TEST_CASE("test_complex_matmul") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<complex>{R(2), R(3)},
                               std::vector<tcapint>{1, 2}, false, TEST_DTAG);
  TensorPtr y =
      std::make_shared<Tensor>(std::vector<complex>{R(4), R(5)},
                               std::vector<tcapint>{2, 1}, false, TEST_DTAG);
  TensorPtr z = x >> y;

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), R(23));

  x = std::make_shared<Tensor>(std::vector<complex>{R(2), R(3)},
                               std::vector<tcapint>{2, 1}, false, TEST_DTAG);
  y = std::make_shared<Tensor>(std::vector<complex>{R(4), R(5)},
                               std::vector<tcapint>{1, 2}, false, TEST_DTAG);
  z = x >> y;

  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[0])[0]), R(8));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[0])[1]), R(12));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[1])[0]), R(10));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[1])[1]), R(15));
}

TEST_CASE("test_mixed_matmul") {
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(2), R(3)},
                               std::vector<tcapint>{1, 2}, false, TEST_DTAG);
  TensorPtr y =
      std::make_shared<Tensor>(std::vector<complex>{R(4), R(5)},
                               std::vector<tcapint>{2, 1}, false, TEST_DTAG);
  TensorPtr z = x >> y;

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), R(23));

  x = std::make_shared<Tensor>(std::vector<complex>{R(2), R(3)},
                               std::vector<tcapint>{1, 2}, false, TEST_DTAG);
  y = std::make_shared<Tensor>(std::vector<real1>{R(4), R(5)},
                               std::vector<tcapint>{2, 1}, false, TEST_DTAG);
  z = y << x;

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), R(23));

  REQUIRE_CMPLX(GET_COMPLEX((*(z.get()))[0]), R(23));

  x = std::make_shared<Tensor>(std::vector<real1>{R(2), R(3)},
                               std::vector<tcapint>{2, 1}, false, TEST_DTAG);
  y = std::make_shared<Tensor>(std::vector<complex>{R(4), R(5)},
                               std::vector<tcapint>{1, 2}, false, TEST_DTAG);
  z = y << x;

  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[0])[0]), R(8));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[0])[1]), R(12));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[1])[0]), R(10));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[1])[1]), R(15));

  x = std::make_shared<Tensor>(std::vector<complex>{R(2), R(3)},
                               std::vector<tcapint>{2, 1}, false, TEST_DTAG);
  y = std::make_shared<Tensor>(std::vector<real1>{R(4), R(5)},
                               std::vector<tcapint>{1, 2}, false, TEST_DTAG);
  z = x >> y;

  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[0])[0]), R(8));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[0])[1]), R(12));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[1])[0]), R(10));
  REQUIRE_CMPLX(GET_COMPLEX((*(*(z.get()))[1])[1]), R(15));
}

TEST_CASE("test_softmax_forward_sums_to_one") {
  // softmax output must be a probability distribution (sum == 1)
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, false, TEST_DTAG);
  TensorPtr y = Tensor::softmax(x, 0);

  RealStorage *ys = static_cast<RealStorage *>(y->storage.get());
  real1_f sum = (real1_f)(*ys)[0] + (real1_f)(*ys)[1] + (real1_f)(*ys)[2];
  REQUIRE_FLOAT(sum, 1.0f);
}

TEST_CASE("test_softmax_forward_order_preserving") {
  // larger input => larger output probability
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(3), R(2)},
                               std::vector<tcapint>{3}, false, TEST_DTAG);
  TensorPtr y = Tensor::softmax(x, 0);

  RealStorage *ys = static_cast<RealStorage *>(y->storage.get());
  REQUIRE((real1_f)(*ys)[1] > (real1_f)(*ys)[2]);
  REQUIRE((real1_f)(*ys)[2] > (real1_f)(*ys)[0]);
}

TEST_CASE("test_softmax_forward_numerical_stability") {
  // large inputs must not produce inf/nan
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1000), R(1001), R(1002)},
                               std::vector<tcapint>{3}, false, TEST_DTAG);
  TensorPtr y = Tensor::softmax(x, 0);

  RealStorage *ys = static_cast<RealStorage *>(y->storage.get());
  real1_f sum = (real1_f)(*ys)[0] + (real1_f)(*ys)[1] + (real1_f)(*ys)[2];
  REQUIRE(std::isfinite(sum));
  REQUIRE_FLOAT(sum, 1.0f);
}

TEST_CASE("test_softmax_forward_uniform_input") {
  // equal inputs => equal outputs
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(2), R(2), R(2)},
                               std::vector<tcapint>{3}, false, TEST_DTAG);
  TensorPtr y = Tensor::softmax(x, 0);

  RealStorage *ys = static_cast<RealStorage *>(y->storage.get());
  REQUIRE_FLOAT((real1_f)(*ys)[0], 1.0f / 3.0f);
  REQUIRE_FLOAT((real1_f)(*ys)[1], 1.0f / 3.0f);
  REQUIRE_FLOAT((real1_f)(*ys)[2], 1.0f / 3.0f);
}

TEST_CASE("test_softmax_forward_known_values") {
  // softmax([0, 1]) = [e^0/(e^0+e^1), e^1/(e^0+e^1)]
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(0), R(1)},
                               std::vector<tcapint>{2}, false, TEST_DTAG);
  TensorPtr y = Tensor::softmax(x, 0);

  RealStorage *ys = static_cast<RealStorage *>(y->storage.get());
  const real1_f denom = 1.0f + std::exp(1.0f);
  REQUIRE_FLOAT((real1_f)(*ys)[0], 1.0f / denom);
  REQUIRE_FLOAT((real1_f)(*ys)[1], std::exp(1.0f) / denom);
}

TEST_CASE("test_softmax_backward_gradient_sum_zero") {
  // sum of softmax input gradients == 0 (Jacobian row sums to zero)
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::softmax(x, 0);
  TensorPtr loss = Tensor::sum(y);
  Tensor::backward(loss);

  RealStorage *xg = static_cast<RealStorage *>(x->grad->storage.get());
  real1_f gsum = (real1_f)(*xg)[0] + (real1_f)(*xg)[1] + (real1_f)(*xg)[2];
  REQUIRE_FLOAT(gsum, 0.0f);
}

TEST_CASE("test_softmax_backward_gradient_values") {
  // for loss = sum(softmax(x)), gradient = softmax(x) * (1 - softmax(x))
  // but summed: dx_i = s_i - s_i * sum(s) = s_i - s_i = 0 already covered
  // use loss = softmax(x)[1] to get a non-trivial gradient
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{R(0), R(0)}, std::vector<tcapint>{2}, true, TEST_DTAG);
  TensorPtr y = Tensor::softmax(x, 0);
  // pick element [1] as scalar loss via slice
  TensorPtr loss = Tensor::sum(Tensor::slice(y, 0, 1, 1));
  Tensor::backward(loss);

  // softmax([0,0]) = [0.5, 0.5]
  // d/dx_0 of s_1 = -s_0 * s_1 = -0.25
  // d/dx_1 of s_1 =  s_1*(1-s_1) = 0.25
  RealStorage *xg = static_cast<RealStorage *>(x->grad->storage.get());
  REQUIRE_FLOAT((real1_f)(*xg)[0], -0.25f);
  REQUIRE_FLOAT((real1_f)(*xg)[1], 0.25f);
}

TEST_CASE("test_softmax_axis1") {
  // 2x3 matrix, softmax along axis 1 (each row sums to 1)
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{R(1), R(2), R(3), R(4), R(5), R(6)},
      std::vector<tcapint>{2, 3}, false, TEST_DTAG);
  TensorPtr y = Tensor::softmax(x, 1);

  RealStorage *ys = static_cast<RealStorage *>(y->storage.get());
  // col-major layout: element [r, c] is at index r + c*nrows
  // row 0: elements at indices 0, 2, 4
  real1_f row0_sum = (real1_f)(*ys)[0] + (real1_f)(*ys)[2] + (real1_f)(*ys)[4];
  // row 1: elements at indices 1, 3, 5
  real1_f row1_sum = (real1_f)(*ys)[1] + (real1_f)(*ys)[3] + (real1_f)(*ys)[5];
  REQUIRE_FLOAT(row0_sum, 1.0f);
  REQUIRE_FLOAT(row1_sum, 1.0f);
}

TEST_CASE("test_logsoftmax_forward_known_values") {
  // logsoftmax([0,1]) = log(softmax([0,1]))
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(0), R(1)},
                               std::vector<tcapint>{2}, false, TEST_DTAG);
  TensorPtr y = Tensor::logsoftmax(x, 0);

  RealStorage *ys = static_cast<RealStorage *>(y->storage.get());
  const real1_f denom = 1.0f + std::exp(1.0f);
  REQUIRE_FLOAT((real1_f)(*ys)[0], std::log(1.0f / denom));
  REQUIRE_FLOAT((real1_f)(*ys)[1], std::log(std::exp(1.0f) / denom));
}

TEST_CASE("test_logsoftmax_forward_all_nonpositive") {
  // log-probabilities must all be <= 0
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, false, TEST_DTAG);
  TensorPtr y = Tensor::logsoftmax(x, 0);

  RealStorage *ys = static_cast<RealStorage *>(y->storage.get());
  REQUIRE((real1_f)(*ys)[0] <= 0.0f);
  REQUIRE((real1_f)(*ys)[1] <= 0.0f);
  REQUIRE((real1_f)(*ys)[2] <= 0.0f);
}

TEST_CASE("test_logsoftmax_forward_matches_log_softmax") {
  // logsoftmax(x) must equal log(softmax(x)) element-wise
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, false, TEST_DTAG);
  TensorPtr y_ls = Tensor::logsoftmax(x, 0);
  TensorPtr y_s = Tensor::softmax(x, 0);
  TensorPtr y_log_s = Tensor::log(y_s);

  RealStorage *ls = static_cast<RealStorage *>(y_ls->storage.get());
  RealStorage *log_s = static_cast<RealStorage *>(y_log_s->storage.get());
  REQUIRE_FLOAT((real1_f)(*ls)[0], (real1_f)(*log_s)[0]);
  REQUIRE_FLOAT((real1_f)(*ls)[1], (real1_f)(*log_s)[1]);
  REQUIRE_FLOAT((real1_f)(*ls)[2], (real1_f)(*log_s)[2]);
}

TEST_CASE("test_logsoftmax_forward_numerical_stability") {
  // large inputs must not produce inf/nan
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1000), R(1001), R(1002)},
                               std::vector<tcapint>{3}, false, TEST_DTAG);
  TensorPtr y = Tensor::logsoftmax(x, 0);

  RealStorage *ys = static_cast<RealStorage *>(y->storage.get());
  REQUIRE(std::isfinite((real1_f)(*ys)[0]));
  REQUIRE(std::isfinite((real1_f)(*ys)[1]));
  REQUIRE(std::isfinite((real1_f)(*ys)[2]));
}

TEST_CASE("test_logsoftmax_backward_gradient_sum") {
  // for loss = sum(logsoftmax(x)), dx_i = 1 - n*softmax(x)_i
  // so sum(dx) = n - n = 0
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(3)},
                               std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr y = Tensor::logsoftmax(x, 0);
  TensorPtr loss = Tensor::sum(y);
  Tensor::backward(loss);

  RealStorage *xg = static_cast<RealStorage *>(x->grad->storage.get());
  real1_f gsum = (real1_f)(*xg)[0] + (real1_f)(*xg)[1] + (real1_f)(*xg)[2];
  REQUIRE_FLOAT(gsum, 0.0f);
}

TEST_CASE("test_logsoftmax_backward_gradient_values") {
  // loss = logsoftmax([0,0])[0]
  // d/dx_0 of lsm_0 = 1 - s_0 =  0.5
  // d/dx_1 of lsm_0 =   - s_1 = -0.5
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{R(0), R(0)}, std::vector<tcapint>{2}, true, TEST_DTAG);
  TensorPtr y = Tensor::logsoftmax(x, 0);
  TensorPtr loss = Tensor::sum(Tensor::slice(y, 0, 0, 1));
  Tensor::backward(loss);

  RealStorage *xg = static_cast<RealStorage *>(x->grad->storage.get());
  REQUIRE_FLOAT((real1_f)(*xg)[0], 0.5f);
  REQUIRE_FLOAT((real1_f)(*xg)[1], -0.5f);
}

TEST_CASE("test_logsoftmax_axis1") {
  // 2x3 matrix, logsoftmax along axis 1; each row's exp-sum must == 1
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{R(1), R(2), R(3), R(4), R(5), R(6)},
      std::vector<tcapint>{2, 3}, false, TEST_DTAG);
  TensorPtr y = Tensor::logsoftmax(x, 1);

  RealStorage *ys = static_cast<RealStorage *>(y->storage.get());
  // col-major: row 0 at indices 0, 2, 4; row 1 at indices 1, 3, 5
  real1_f row0_expsum = std::exp((real1_f)(*ys)[0]) +
                        std::exp((real1_f)(*ys)[2]) +
                        std::exp((real1_f)(*ys)[4]);
  real1_f row1_expsum = std::exp((real1_f)(*ys)[1]) +
                        std::exp((real1_f)(*ys)[3]) +
                        std::exp((real1_f)(*ys)[5]);
  REQUIRE_FLOAT(row0_expsum, 1.0f);
  REQUIRE_FLOAT(row1_expsum, 1.0f);
}

TEST_CASE("test_matmul_gradient_sum_loss") {
  using namespace Weed;

  // A: 2x3 (column-major)
  TensorPtr A = std::make_shared<Tensor>(
      std::vector<real1>{R(1), R(2), R(3), R(4), R(5), R(6)},
      std::vector<tcapint>{2, 3}, true, TEST_DTAG);

  // B: 3x2 (column-major)
  TensorPtr B = std::make_shared<Tensor>(
      std::vector<real1>{R(7), R(8), R(9), R(10), R(11), R(12)},
      std::vector<tcapint>{3, 2}, true, TEST_DTAG);

  TensorPtr C = A >> B;
  TensorPtr L = Tensor::sum(C);

  Tensor::backward(L);

  RealStorage *Ag = static_cast<RealStorage *>(A->grad->storage.get());
  REQUIRE((*Ag)[0] == R(17));
  REQUIRE((*Ag)[1] == R(17));
  REQUIRE((*Ag)[2] == R(19));
  REQUIRE((*Ag)[3] == R(19));
  REQUIRE((*Ag)[4] == R(21));
  REQUIRE((*Ag)[5] == R(21));

  RealStorage *Bg = static_cast<RealStorage *>(B->grad->storage.get());
  REQUIRE((*Bg)[0] == R(3));
  REQUIRE((*Bg)[1] == R(7));
  REQUIRE((*Bg)[2] == R(11));
  REQUIRE((*Bg)[3] == R(3));
  REQUIRE((*Bg)[4] == R(7));
  REQUIRE((*Bg)[5] == R(11));
}
