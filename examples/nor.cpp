//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of
// Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#include "tensors/symbol_tensor.hpp"

#include "autograd/adam.hpp"
// #include "autograd/bci_loss.hpp"
#include "autograd/mse_loss.hpp"
// #include "autograd/sgd.hpp"
#include "autograd/zero_grad.hpp"
#include "modules/linear.hpp"
#include "tensors/real_scalar.hpp"

#include <iostream> // For cout

#define GET_REAL(ptr) static_cast<RealScalar *>((ptr).get())->get_item()

#define R(v) real1(v)
#define C(v) complex(v)

using namespace Weed;

int main() {
  TensorPtr x = std::make_shared<Tensor>(
      std::vector<real1>{R(0), R(1), R(0), R(1), R(0), R(0), R(1), R(1)},
      std::vector<tcapint>{4, 2}, std::vector<tcapint>{1, 4}, false,
      DeviceTag::CPU);
  TensorPtr y = std::make_shared<Tensor>(
      std::vector<real1>{R(1), R(0), R(0), R(0)}, std::vector<tcapint>{4, 1},
      std::vector<tcapint>{1, 0}, false, DeviceTag::CPU);

  Linear l1(2, 4, true, DType::REAL, DeviceTag::CPU);
  Linear l2(4, 1, true, DType::REAL, DeviceTag::CPU);
  Linear l3(1, 1, true, DType::REAL, DeviceTag::CPU);

  std::vector<ParameterPtr> params = l1.parameters();
  std::vector<ParameterPtr> params2 = l2.parameters();
  std::vector<ParameterPtr> params3 = l3.parameters();
  params.insert(params.begin(), params2.begin(), params2.end());
  params.insert(params.begin(), params3.begin(), params3.end());

  Adam opt(R(0.05));
  opt.register_parameters(params);

  size_t epoch = 1;
  real1 loss_r = ONE_R1;

  while ((epoch <= 10) && (loss_r > 0.1)) {
    TensorPtr y_pred = Tensor::sigmoid(
        l3.forward(Tensor::sigmoid(l2.forward(Tensor::relu(l1.forward(x))))));
    // TensorPtr loss = bci_loss(y_pred, y);
    TensorPtr loss = mse_loss(y_pred, y);

    Tensor::backward(loss);
    adam_step(opt, params);
    // sgd_step(params, 1.0);

    loss_r = GET_REAL(loss);
    std::cout << "Epoch " << epoch << ", Loss: " << loss_r << std::endl;

    zero_grad(params);
    ++epoch;
  }

  std::cout << "In: [[0, 0], [1, 0], [0, 1], [1, 1]]" << std::endl;

  TensorPtr y_pred =
      Tensor::sigmoid(l2.forward(Tensor::sigmoid(l1.forward(x))));
  RealStorage &storage = *static_cast<RealStorage *>(y_pred->storage.get());

  std::cout << "Out: [";
  for (size_t i = 0U; i < 4U; ++i) {
    std::cout << "[" << ((storage[i] < 0.5) ? "0" : "1") << "]";
    if (i < 3U) {
      std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
}
