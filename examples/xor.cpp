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

#include "autograd/sgd.hpp"
#include "autograd/zero_grad.hpp"
#include "modules/linear.hpp"

#include <iostream> // For cout

#define GET_REAL(ptr) static_cast<RealScalar *>((ptr).get())->get_item()

using namespace Weed;

int main() {
  TensorPtr x = std::make_shared<Tensor>(
    std::vector<real1>{0, 0, 1, 0, 0, 1, 1, 1},
    std::vector<vecCapInt>{4, 2},
    std::vector<vecCapInt>{2, 1});
  TensorPtr y = std::make_shared<Tensor>(
    std::vector<real1>{0, 1, 1, 0},
    std::vector<vecCapInt>{4, 1},
    std::vector<vecCapInt>{1, 1});

  Linear l1(2, 8);
  Linear l2(8, 1);

  std::vector<ParameterPtr> params = l1.parameters();
  std::vector<ParameterPtr> params2 = l2.parameters();
  params.insert(params.end(), params2.begin(), params2.end());

  size_t epoch = 1;
  real1 loss_r = ONE_R1;

  while ((epoch <= 1000) && (loss_r > 0.01)) {
    TensorPtr y_pred = l2.forward(Tensor::relu(l1.forward(x)));
    TensorPtr loss = Tensor::mean((y_pred - y) * (y_pred - y));

    Tensor::backward(loss);
    sgd_step(params, 0.1);

    loss_r = GET_REAL(loss);
    if (!(epoch % 100)) {
      std::cout << "Epoch " << epoch << ", Loss: " << loss_r << std::endl;
    }

    zero_grad(params);
    ++epoch;
  }
}
