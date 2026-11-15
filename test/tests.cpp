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
#include "modules/embedding.hpp"
#include "modules/learned_positional_encoding.hpp"
#include "modules/linear.hpp"
#include "modules/positional_encoding.hpp"
#include "modules/qwen_decoder_layer.hpp"
#include "modules/sequential.hpp"
#include "modules/transformer_encoder_layer.hpp"
#include "ops/triu_fill.hpp"
#include "storage/all_storage.hpp"
#include "tensors/complex_scalar.hpp"
#include "tensors/real_scalar.hpp"
#include "tensors/real_tensor.hpp"
#include "tensors/symbol_tensor.hpp"

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

// Flat (column-major) read of a real tensor, materializing any view
static std::vector<real1_f> to_vector(TensorPtr t) {
  // (Read on the CPU, whichever device the tensor is on.)
  TensorPtr c = Tensor::contiguous(Tensor::detach(t)->cast(DeviceTag::CPU));
  RealTensor *r = static_cast<RealTensor *>(c.get());
  const tcapint n = c->get_broadcast_size();
  std::vector<real1_f> v(n);
  for (tcapint i = 0U; i < n; ++i) {
    v[i] = (real1_f)(*r)[i];
  }

  return v;
}

TEST_CASE("test_linear_bias_broadcast_is_not_stale") {
  // A bias that was once broadcast against (1, 4, out) must not force a later
  // (1, 1, in) input to be broadcast up to (1, 4, out).
  LinearPtr l =
      std::make_shared<Linear>(2, 3, true, true, DType::REAL, TEST_DTAG);
  l->eval();
  TensorPtr x4 = std::make_shared<Tensor>(
      std::vector<real1>{R(1), R(2), R(3), R(4), R(5), R(6), R(7), R(8)},
      std::vector<tcapint>{1, 4, 2}, false, TEST_DTAG);
  TensorPtr y4 = l->forward(x4);
  REQUIRE(y4->shape == std::vector<tcapint>{1, 4, 3});

  // Same values as token 0 of x4
  TensorPtr x1 =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(5)},
                               std::vector<tcapint>{1, 1, 2}, false, TEST_DTAG);
  TensorPtr y1 = l->forward(x1);
  REQUIRE(y1->shape == std::vector<tcapint>{1, 1, 3});

  const std::vector<real1_f> v4 = to_vector(y4);
  const std::vector<real1_f> v1 = to_vector(y1);
  for (tcapint o = 0U; o < 3U; ++o) {
    REQUIRE_FLOAT(v1[o], v4[4U * o]);
  }
}

TEST_CASE("test_positional_encoding_values") {
  const tcapint max_len = 5U;
  const tcapint d_model = 6U;
  const real1_f pos_val = 100.0f;
  PositionalEncoding pe(max_len, d_model, pos_val, TEST_DTAG);

  const tcapint T = 4U;
  TensorPtr x =
      Tensor::zeros({1U, T, d_model}, false, false, DType::REAL, TEST_DTAG);
  const std::vector<real1_f> v = to_vector(pe.forward(x));
  for (tcapint pos = 0U; pos < T; ++pos) {
    for (tcapint i = 0U; i < (d_model >> 1U); ++i) {
      const double angle = pos / std::pow((double)pos_val, (2.0 * i) / d_model);
      // (0, pos, c) is at flat (column-major) index pos + T * c
      REQUIRE_FLOAT(v[pos + T * (2U * i)], (real1_f)std::cos(angle));
      REQUIRE_FLOAT(v[pos + T * (2U * i + 1U)], (real1_f)std::sin(angle));
    }
  }
}

static SymbolTensorPtr token_tensor(const std::vector<symint> &tokens,
                                    const size_t start, const size_t len) {
  return std::make_shared<SymbolTensor>(
      std::vector<symint>(tokens.begin() + start, tokens.begin() + start + len),
      std::vector<tcapint>{1U, (tcapint)len}, false, TEST_DTAG);
}

// Embedding with deterministic, distinct, nonzero rows
static EmbeddingPtr test_embedding(const tcapint vocab, const tcapint d_model) {
  EmbeddingPtr e =
      std::make_shared<Embedding>(vocab, d_model, DType::REAL, TEST_DTAG);
  std::vector<real1> w(vocab * d_model);
  for (tcapint i = 0U; i < (vocab * d_model); ++i) {
    w[i] = (real1)std::sin(0.7 + 1.3 * (double)i);
  }
  e->weight = std::make_shared<Parameter>(
      w, std::vector<tcapint>{vocab, d_model}, TEST_DTAG);

  return e;
}

// Feed "tokens" in chunks through the KV cache, and compare every position's
// output with a single full-sequence pass.
static void check_kv_cache_matches_full(Sequential &model,
                                        const std::vector<symint> &tokens,
                                        const std::vector<size_t> &chunks,
                                        const tcapint out_dim) {
  const tcapint T = (tcapint)tokens.size();
  model.eval();
  model.reset_cache();
  const std::vector<real1_f> full =
      to_vector(model.forward(token_tensor(tokens, 0U, tokens.size())));
  REQUIRE(full.size() == (size_t)(T * out_dim));
  REQUIRE(model.get_kv_cache_len() == T);

  model.reset_cache();
  REQUIRE(model.get_kv_cache_len() == 0U);
  size_t start = 0U;
  for (const size_t len : chunks) {
    TensorPtr y = model.forward(token_tensor(tokens, start, len));
    REQUIRE(y->shape == std::vector<tcapint>{1U, (tcapint)len, out_dim});
    const std::vector<real1_f> part = to_vector(y);
    for (size_t t = 0U; t < len; ++t) {
      for (tcapint o = 0U; o < out_dim; ++o) {
        REQUIRE_FLOAT(part[t + len * o], full[(start + t) + T * o]);
      }
    }
    start += len;
  }
  REQUIRE(start == tokens.size());
  model.reset_cache();
}

TEST_CASE("test_transformer_kv_cache_matches_full_pass") {
  const tcapint vocab = 7U, d_model = 8U, d_ff = 16U, heads = 2U, max_len = 12U;
  std::vector<ModulePtr> layers{
      test_embedding(vocab, d_model),
      std::make_shared<LearnedPositionalEncoding>(max_len, d_model, TEST_DTAG)};
  for (int i = 0; i < 2; ++i) {
    TransformerEncoderLayerPtr l = std::make_shared<TransformerEncoderLayer>(
        d_model, heads, d_ff, TEST_DTAG);
    l->self_attn->kv_quant_bits = 0; // exact KV cache
    layers.push_back(l);
  }
  layers.push_back(std::make_shared<Linear>(d_model, vocab, true, true,
                                            DType::REAL, TEST_DTAG));
  Sequential model(layers);

  const std::vector<symint> tokens{3, 1, 4, 1, 5, 2, 6, 0};
  // One token at a time after a prompt (like text generation)
  check_kv_cache_matches_full(model, tokens, {3U, 1U, 1U, 1U, 1U, 1U}, vocab);
  // One token at a time from the start
  check_kv_cache_matches_full(model, tokens, {1U, 1U, 1U, 1U, 1U, 1U, 1U, 1U},
                              vocab);
  // Several tokens at a time, continuing from a non-empty cache
  check_kv_cache_matches_full(model, tokens, {2U, 4U, 2U}, vocab);

  // The KV cache is bypassed in training mode
  model.train();
  TensorPtr y = model.forward(token_tensor(tokens, 0U, 4U));
  REQUIRE(model.get_kv_cache_len() == 0U);
  y = model.forward(token_tensor(tokens, 0U, 4U));
  REQUIRE(y->shape == std::vector<tcapint>{1U, 4U, vocab});
  model.eval();
}

TEST_CASE("test_qwen_kv_cache_matches_full_pass") {
  const tcapint vocab = 7U, d_model = 8U, d_ff = 16U, heads = 4U, kv_heads = 2U;
  std::vector<ModulePtr> layers{test_embedding(vocab, d_model)};
  for (int i = 0; i < 2; ++i) {
    QwenDecoderLayerPtr l =
        std::make_shared<QwenDecoderLayer>(d_model, heads, kv_heads, d_ff, 16U);
    l->self_attn->kv_quant_bits = 0; // exact KV cache
    layers.push_back(l);
  }
  layers.push_back(std::make_shared<Linear>(d_model, vocab, false, true,
                                            DType::REAL, TEST_DTAG));
  Sequential model(layers);

  const std::vector<symint> tokens{3, 1, 4, 1, 5, 2, 6, 0};
  check_kv_cache_matches_full(model, tokens, {3U, 1U, 1U, 1U, 1U, 1U}, vocab);
  check_kv_cache_matches_full(model, tokens, {2U, 4U, 2U}, vocab);
}

TEST_CASE("test_transformer_kv_cache_overflow_throws") {
  const tcapint vocab = 5U, d_model = 4U;
  TransformerEncoderLayerPtr l =
      std::make_shared<TransformerEncoderLayer>(d_model, 1U, 8U, TEST_DTAG);
  Sequential model({test_embedding(vocab, d_model), l});
  model.eval();
  model.set_max_kv_seq_len(3U);
  const std::vector<symint> tokens{1, 2, 3, 4};
  model.forward(token_tensor(tokens, 0U, 3U));
  REQUIRE_THROWS(model.forward(token_tensor(tokens, 3U, 1U)));
  model.reset_cache();
}

TEST_CASE("test_real_matmul_shapes_and_storage") {
  // Row vectors, column vectors, and sparse operands all have to give the
  // same answer as the general case (whichever kernel handles them).
  auto dense = [](const tcapint r, const tcapint c, const double seed) {
    std::vector<real1> v(r * c);
    for (tcapint i = 0U; i < (r * c); ++i) {
      v[i] = (real1)std::sin(seed + 0.9 * (double)i);
    }
    return std::make_shared<Tensor>(v, std::vector<tcapint>{r, c}, false,
                                    TEST_DTAG);
  };
  auto check = [](TensorPtr a, TensorPtr b) {
    const tcapint M = a->shape[0U], K = a->shape[1U], N = b->shape[1U];
    const std::vector<real1_f> av = to_vector(a);
    const std::vector<real1_f> bv = to_vector(b);
    TensorPtr c = a >> b;
    REQUIRE(c->shape == std::vector<tcapint>{M, N});
    const std::vector<real1_f> cv = to_vector(c);
    for (tcapint i = 0U; i < M; ++i) {
      for (tcapint j = 0U; j < N; ++j) {
        real1_f e = 0;
        for (tcapint k = 0U; k < K; ++k) {
          e += av[i + M * k] * bv[k + K * j];
        }
        REQUIRE_FLOAT(cv[i + M * j], e);
      }
    }
  };

  check(dense(3U, 4U, 0.1), dense(4U, 5U, 0.2)); // general
  check(dense(1U, 4U, 0.3), dense(4U, 5U, 0.4)); // single row
  check(dense(3U, 4U, 0.5), dense(4U, 1U, 0.6)); // single column
  check(dense(3U, 1U, 0.7), dense(1U, 5U, 0.8)); // outer product
  check(dense(1U, 4U, 0.9), dense(4U, 1U, 1.0)); // inner product

  // Sparse right-hand operand (as for a zero-initialized Linear weight)
  RealSparseVector sv;
  sv[1U] = R(2);
  sv[7U] = R(-3);
  TensorPtr sparse =
      std::make_shared<Tensor>(sv, std::vector<tcapint>{4U, 2U}, false);
  check(dense(3U, 4U, 1.1), sparse);
}

TEST_CASE("test_axis_sum_3d") {
  // x(i, j, k) = 100 i + 10 j + k
  const tcapint A = 2U, B = 3U, C = 4U;
  std::vector<real1> v(A * B * C);
  for (tcapint i = 0U; i < A; ++i) {
    for (tcapint j = 0U; j < B; ++j) {
      for (tcapint k = 0U; k < C; ++k) {
        v[i + A * (j + B * k)] = R(100 * i + 10 * j + k);
      }
    }
  }
  const tcapint dims[3U] = {A, B, C};
  for (tcapint axis = 0U; axis < 3U; ++axis) {
    TensorPtr x = std::make_shared<Tensor>(v, std::vector<tcapint>{A, B, C},
                                           true, TEST_DTAG);
    TensorPtr y = Tensor::sum(x, axis);
    tcapint rd[3U] = {A, B, C};
    rd[axis] = 1U;
    REQUIRE(y->shape == std::vector<tcapint>{rd[0U], rd[1U], rd[2U]});
    const std::vector<real1_f> yv = to_vector(y);
    for (tcapint i = 0U; i < rd[0U]; ++i) {
      for (tcapint j = 0U; j < rd[1U]; ++j) {
        for (tcapint k = 0U; k < rd[2U]; ++k) {
          real1_f e = 0;
          for (tcapint q = 0U; q < dims[axis]; ++q) {
            e += 100 * ((axis == 0U) ? q : i) + 10 * ((axis == 1U) ? q : j) +
                 ((axis == 2U) ? q : k);
          }
          REQUIRE_FLOAT(yv[i + rd[0U] * (j + rd[1U] * k)], e);
        }
      }
    }

    // d/dx of a weighted sum of y: every x(i, j, k) gets the weight of the
    // element of y that it was summed into.
    std::vector<real1> w(yv.size());
    for (size_t n = 0U; n < w.size(); ++n) {
      w[n] = R(1 + n);
    }
    TensorPtr wt = std::make_shared<Tensor>(w, y->shape, false, TEST_DTAG);
    Tensor::backward(Tensor::sum(y * wt));
    const std::vector<real1_f> g = to_vector(Tensor::detach(x->grad));
    for (tcapint i = 0U; i < A; ++i) {
      for (tcapint j = 0U; j < B; ++j) {
        for (tcapint k = 0U; k < C; ++k) {
          const tcapint yi = (axis == 0U) ? 0U : i;
          const tcapint yj = (axis == 1U) ? 0U : j;
          const tcapint yk = (axis == 2U) ? 0U : k;
          REQUIRE_FLOAT(g[i + A * (j + B * k)],
                        w[yi + rd[0U] * (yj + rd[1U] * yk)]);
        }
      }
    }
  }
}

TEST_CASE("test_view_gradients") {
  const std::vector<real1> v{R(1), R(2), R(3), R(4), R(5), R(6)};
  const std::vector<real1> wv{R(1),    R(10),    R(100),
                              R(1000), R(10000), R(100000)};

  // reshape: same elements, same order
  {
    TensorPtr x = std::make_shared<Tensor>(v, std::vector<tcapint>{2, 3}, true,
                                           TEST_DTAG);
    TensorPtr y = Tensor::reshape(x * R(1), {3, 2});
    Tensor::backward(Tensor::sum(y * y));
    const std::vector<real1_f> g = to_vector(Tensor::detach(x->grad));
    for (size_t i = 0U; i < 6U; ++i) {
      REQUIRE_FLOAT(g[i], 2 * v[i]);
    }
  }

  // transpose: x(i, j) lands at y(j, i)
  {
    TensorPtr x = std::make_shared<Tensor>(v, std::vector<tcapint>{2, 3}, true,
                                           TEST_DTAG);
    TensorPtr y = Tensor::transpose(x * R(1), 0, 1);
    TensorPtr w = std::make_shared<Tensor>(wv, std::vector<tcapint>{3, 2},
                                           false, TEST_DTAG);
    Tensor::backward(Tensor::sum(y * w));
    const std::vector<real1_f> g = to_vector(Tensor::detach(x->grad));
    for (tcapint i = 0U; i < 2U; ++i) {
      for (tcapint j = 0U; j < 3U; ++j) {
        REQUIRE_FLOAT(g[i + 2U * j], wv[j + 3U * i]);
      }
    }
  }

  // batched matmul: L = sum(x @ x^T) over a batch of two 3 x 1... matrices
  {
    TensorPtr x = std::make_shared<Tensor>(v, std::vector<tcapint>{2, 3, 1},
                                           true, TEST_DTAG);
    TensorPtr xs = x * R(1);
    TensorPtr y = xs >> Tensor::transpose(xs, -2, -1); // (2, 3, 3)
    Tensor::backward(Tensor::sum(y));
    // d/dx_b[i] sum_{p,q} x_b[p] x_b[q] = 2 sum_p x_b[p]
    const std::vector<real1_f> g = to_vector(Tensor::detach(x->grad));
    for (tcapint b = 0U; b < 2U; ++b) {
      const real1_f s = v[b] + v[b + 2U] + v[b + 4U];
      for (tcapint i = 0U; i < 3U; ++i) {
        REQUIRE_FLOAT(g[b + 2U * i], 2 * s);
      }
    }
  }
}

TEST_CASE("test_div_denominator_gradient_chain_rule") {
  // L = sum(w * (a / b)), so dL/db = -w * a / b^2
  const std::vector<real1> av{R(1), R(2), R(3)};
  const std::vector<real1> bv{R(2), R(4), R(5)};
  const std::vector<real1> wv{R(3), R(-1), R(10)};
  TensorPtr a =
      std::make_shared<Tensor>(av, std::vector<tcapint>{3}, false, TEST_DTAG);
  TensorPtr b =
      std::make_shared<Tensor>(bv, std::vector<tcapint>{3}, true, TEST_DTAG);
  TensorPtr w =
      std::make_shared<Tensor>(wv, std::vector<tcapint>{3}, false, TEST_DTAG);
  Tensor::backward(Tensor::sum((a / b) * w));
  const std::vector<real1_f> g = to_vector(Tensor::detach(b->grad));
  for (size_t i = 0U; i < 3U; ++i) {
    REQUIRE_FLOAT(g[i], -wv[i] * av[i] / (bv[i] * bv[i]));
  }
}

TEST_CASE("test_sgd_broadcast_bias_update") {
  // The bias is broadcast over 4 sequence positions in the forward pass, but
  // an SGD step must still only apply its (summed) gradient once.
  LinearPtr l =
      std::make_shared<Linear>(2, 3, true, true, DType::REAL, TEST_DTAG);
  l->train();
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{R(1), R(2), R(3), R(4), R(5), R(6), R(7), R(8)},
      std::vector<tcapint>{1, 4, 2}, false, TEST_DTAG);
  Tensor::backward(Tensor::sum(l->forward(x)));
  // d(sum)/d(bias[o]) = 4 (one per position), and the bias starts at 0.
  sgd_step(l->parameters(), R(0.5));
  l->bias->reset_broadcast();
  const std::vector<real1_f> bias = to_vector(Tensor::detach(l->bias));
  REQUIRE(bias.size() == 3U);
  for (size_t o = 0U; o < 3U; ++o) {
    REQUIRE_FLOAT(bias[o], -2.0f);
  }
}

// The tests below use shapes that the element-wise GPU kernels can't address
// with a single step (or that used to be mistaken for scalars), so they are
// most useful when small tensors are forced onto the device:
//     WEED_GSTRIDE=1 ./unittest --device-gpu

TEST_CASE("test_leading_extent_one_and_broadcast") {
  // (1, 3, 4): a contiguous tensor whose stride[0] is 0
  const tcapint T = 3U, D = 4U;
  std::vector<real1> xv(T * D);
  for (tcapint i = 0U; i < (T * D); ++i) {
    xv[i] = (real1)std::sin(0.3 + 1.37 * (double)i);
  }
  const std::vector<real1> bv{R(10), R(20), R(30), R(40)};

  TensorPtr x = std::make_shared<Tensor>(xv, std::vector<tcapint>{1U, T, D},
                                         false, TEST_DTAG);
  TensorPtr b =
      std::make_shared<Tensor>(bv, std::vector<tcapint>{D}, false, TEST_DTAG);

  const std::vector<real1_f> relu = to_vector(Tensor::relu(x));
  const std::vector<real1_f> tanh = to_vector(Tensor::tanh(x));
  const std::vector<real1_f> sq = to_vector(x * x);
  const std::vector<real1_f> xb = to_vector(x + b);
  const std::vector<real1_f> xg = to_vector(x * b);
  // transposed view: y(0, d, t) = x(0, t, d)
  const std::vector<real1_f> xt = to_vector(Tensor::transpose(x, 1, 2) * R(2));
  for (tcapint t = 0U; t < T; ++t) {
    for (tcapint d = 0U; d < D; ++d) {
      const real1_f v = xv[t + T * d];
      REQUIRE_FLOAT(relu[t + T * d], (v > 0) ? v : 0);
      REQUIRE_FLOAT(tanh[t + T * d], (real1_f)std::tanh(v));
      REQUIRE_FLOAT(sq[t + T * d], v * v);
      REQUIRE_FLOAT(xb[t + T * d], v + bv[d]);
      REQUIRE_FLOAT(xg[t + T * d], v * bv[d]);
      REQUIRE_FLOAT(xt[d + D * t], 2 * v);
    }
  }
}

TEST_CASE("test_softmax_middle_axis") {
  const tcapint A = 2U, B = 3U, C = 4U;
  std::vector<real1> xv(A * B * C);
  for (tcapint i = 0U; i < (A * B * C); ++i) {
    xv[i] = (real1)std::sin(0.9 + 0.71 * (double)i);
  }
  TensorPtr x = std::make_shared<Tensor>(xv, std::vector<tcapint>{A, B, C},
                                         false, TEST_DTAG);
  const std::vector<real1_f> y = to_vector(Tensor::softmax(x, 1));
  const std::vector<real1_f> ly = to_vector(Tensor::logsoftmax(x, 1));
  for (tcapint i = 0U; i < A; ++i) {
    for (tcapint k = 0U; k < C; ++k) {
      real1_f z = 0;
      for (tcapint j = 0U; j < B; ++j) {
        z += (real1_f)std::exp(xv[i + A * (j + B * k)]);
      }
      for (tcapint j = 0U; j < B; ++j) {
        const tcapint n = i + A * (j + B * k);
        REQUIRE_FLOAT(y[n], (real1_f)std::exp(xv[n]) / z);
        REQUIRE_FLOAT(ly[n], xv[n] - (real1_f)std::log(z));
      }
    }
  }
}

TEST_CASE("test_triu_fill_nonsquare") {
  // 2 rows, 5 columns, from the 2nd superdiagonal: element (row, col) is
  // filled if row + 2 <= col.
  const tcapint rows = 2U, cols = 5U;
  TensorPtr m =
      Tensor::zeros({rows, cols}, false, false, DType::REAL, TEST_DTAG);
  Weed::triu_fill(*m, R(-7), 2U);
  const std::vector<real1_f> v = to_vector(m);
  for (tcapint r = 0U; r < rows; ++r) {
    for (tcapint c = 0U; c < cols; ++c) {
      REQUIRE_FLOAT(v[r + rows * c], ((r + 2U) <= c) ? -7.0f : 0.0f);
    }
  }
}

TEST_CASE("test_embedding_repeated_tokens") {
  // Token 1 appears twice, so its row of the gradient gets two contributions.
  const tcapint vocab = 4U, dim = 3U;
  EmbeddingPtr e = test_embedding(vocab, dim);
  e->train();
  const std::vector<symint> ids{1, 0, 3, 1, 2};
  const tcapint N = (tcapint)ids.size();
  SymbolTensorPtr idx = std::make_shared<SymbolTensor>(
      ids, std::vector<tcapint>{1U, N}, false, TEST_DTAG);
  TensorPtr y = e->forward(idx);
  REQUIRE(y->shape == std::vector<tcapint>{1U, N, dim});

  const std::vector<real1_f> w = to_vector(Tensor::detach(e->weight));
  const std::vector<real1_f> yv = to_vector(y);
  for (tcapint n = 0U; n < N; ++n) {
    for (tcapint d = 0U; d < dim; ++d) {
      REQUIRE_FLOAT(yv[n + N * d], w[(tcapint)ids[n] + vocab * d]);
    }
  }

  Tensor::backward(Tensor::sum(y));
  const std::vector<real1_f> g = to_vector(Tensor::detach(e->weight->grad));
  const real1_f count[4U] = {1, 2, 1, 1};
  for (tcapint tok = 0U; tok < vocab; ++tok) {
    for (tcapint d = 0U; d < dim; ++d) {
      REQUIRE_FLOAT(g[tok + vocab * d], count[tok]);
    }
  }
}

TEST_CASE("test_mixed_matmul_values") {
  // (complex 2 x 3) @ (real 3 x 2), and (real 2 x 3) @ (complex 3 x 2)
  const std::vector<complex> cv{complex(R(1), R(2)),   complex(R(0), R(-1)),
                                complex(R(3), R(0)),   complex(R(-2), R(1)),
                                complex(R(0.5), R(0)), complex(R(1), R(1))};
  const std::vector<real1> rv{R(1), R(2), R(3), R(4), R(5), R(6)};
  TensorPtr ca = std::make_shared<Tensor>(cv, std::vector<tcapint>{2, 3}, false,
                                          TEST_DTAG);
  TensorPtr rb = std::make_shared<Tensor>(rv, std::vector<tcapint>{3, 2}, false,
                                          TEST_DTAG);
  TensorPtr ra = std::make_shared<Tensor>(rv, std::vector<tcapint>{2, 3}, false,
                                          TEST_DTAG);
  TensorPtr cb = std::make_shared<Tensor>(cv, std::vector<tcapint>{3, 2}, false,
                                          TEST_DTAG);

  TensorPtr left = (ca >> rb)->cast(DeviceTag::CPU);
  TensorPtr right = (ra >> cb)->cast(DeviceTag::CPU);
  ComplexStorage *ls = static_cast<ComplexStorage *>(left->storage.get());
  ComplexStorage *rs = static_cast<ComplexStorage *>(right->storage.get());
  for (tcapint i = 0U; i < 2U; ++i) {
    for (tcapint j = 0U; j < 2U; ++j) {
      complex el = ZERO_CMPLX, er = ZERO_CMPLX;
      for (tcapint k = 0U; k < 3U; ++k) {
        el += cv[i + 2U * k] * rv[k + 3U * j];
        er += rv[i + 2U * k] * cv[k + 3U * j];
      }
      REQUIRE_CMPLX((*ls)[i + 2U * j], el);
      REQUIRE_CMPLX((*rs)[i + 2U * j], er);
    }
  }
}

TEST_CASE("test_mean_axis_1d") {
  // The mean over the only index of a 1-D tensor, and its gradient
  TensorPtr x =
      std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(6)},
                               std::vector<tcapint>{3U}, true, TEST_DTAG);
  TensorPtr y = Tensor::mean(x, 0);
  REQUIRE(y->get_broadcast_size() == 1U);
  REQUIRE_FLOAT(to_vector(y)[0U], R(3));

  Tensor::backward(Tensor::sum(y));
  const std::vector<real1_f> g = to_vector(Tensor::detach(x->grad));
  REQUIRE(g.size() == 3U);
  for (tcapint i = 0U; i < 3U; ++i) {
    REQUIRE_FLOAT(g[i], R(R(1) / R(3)));
  }
}

TEST_CASE("test_mean_axis_extent_one") {
  // The mean over an index of extent 1 must leave its argument's shape alone.
  const std::vector<tcapint> shp{3U, 1U};
  TensorPtr x = std::make_shared<Tensor>(std::vector<real1>{R(1), R(2), R(6)},
                                         shp, true, TEST_DTAG);
  TensorPtr m = Tensor::mean(x, -1);
  REQUIRE(x->shape == shp);
  REQUIRE(m->shape == shp);
  const std::vector<real1_f> mv = to_vector(m);
  REQUIRE_FLOAT(mv[0U], R(1));
  REQUIRE_FLOAT(mv[1U], R(2));
  REQUIRE_FLOAT(mv[2U], R(6));

  // x - mean(x) is identically zero here, so its gradient cancels as well.
  TensorPtr d = x - Tensor::mean(x, -1);
  REQUIRE(x->shape == shp);
  REQUIRE(d->shape == shp);
  for (const real1_f &v : to_vector(d)) {
    REQUIRE_FLOAT(v, R(0));
  }
  Tensor::backward(Tensor::sum(d));
  for (const real1_f &v : to_vector(Tensor::detach(x->grad))) {
    REQUIRE_FLOAT(v, R(0));
  }
}
