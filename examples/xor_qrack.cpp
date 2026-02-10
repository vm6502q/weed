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
#include "autograd/bci_loss.hpp"
#include "autograd/zero_grad.hpp"
#include "modules/linear.hpp"
#include "modules/mean_center.hpp"
#include "modules/qrack_neuron_layer.hpp"
#include "modules/sequential.hpp"
#include "modules/sigmoid.hpp"
#include "tensors/real_scalar.hpp"

#include <fstream>
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
      std::vector<real1>{R(0), R(1), R(1), R(0)}, std::vector<tcapint>{4, 1},
      std::vector<tcapint>{1, 0}, false, DeviceTag::CPU);

  const std::vector<ModulePtr> mv = {
      std::make_shared<QrackNeuronLayer>(2, 1, 0, 2, 2, ALT_BELL_GHZ_QFN),
      std::make_shared<MeanCenter>(),
      std::make_shared<Linear>(1, 1, false, false),
      std::make_shared<Sigmoid>()};

  Sequential model(mv);

  std::vector<ParameterPtr> params = model.parameters();

  Adam opt(R(0.1));
  opt.register_parameters(params);

  size_t epoch = 1;
  real1 loss_r = ONE_R1;

  while ((epoch <= 2000) && (loss_r > 0.01)) {
    TensorPtr y_pred = model.forward(x);
    TensorPtr loss = bci_loss(y_pred, y);

    Tensor::backward(loss);
    adam_step(opt, params);

    loss_r = GET_REAL(loss);
    if ((epoch % 100) == 0U) {
      std::cout << "Epoch " << epoch << ", Loss: " << loss_r << std::endl;
    }

    zero_grad(params);
    ++epoch;
  }

  // Can we save to disk?
  std::ofstream o("xor_qrack.qml");
  model.save(o);
  o.close();

  std::ifstream i("xor_qrack.qml");
  ModulePtr m = Module::load(i);
  i.close();

  m->eval();

  std::cout << "In: [[0, 0], [1, 0], [0, 1], [1, 1]]" << std::endl;

  TensorPtr y_pred = m->forward(x);
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
