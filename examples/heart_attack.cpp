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

#include "common/rapidcsv.h"

#include "autograd/adam.hpp"
#include "autograd/bci_loss.hpp"
#include "autograd/zero_grad.hpp"
#include "modules/linear.hpp"
#include "modules/sequential.hpp"
#include "modules/sigmoid.hpp"
#include "modules/tanh.hpp"
#include "tensors/real_scalar.hpp"

#include <fstream>
#include <iostream> // For cout

#define GET_REAL(ptr) static_cast<RealScalar *>((ptr).get())->get_item()

#define R(v) real1(v)
#define C(v) complex(v)

using namespace Weed;

int main() {
  rapidcsv::Document doc("data/Heart_Attack_Data_Set.csv");

  std::vector<real1> features = doc.GetColumn<real1>("age");
  const tcapint row_count = features.size();
  const tcapint col_count = 13U;

  std::vector<real1> tmp = doc.GetColumn<real1>("sex");
  features.insert(features.end(), tmp.begin(), tmp.end()); 
  tmp = doc.GetColumn<real1>("cp");
  features.insert(features.end(), tmp.begin(), tmp.end());
  tmp = doc.GetColumn<real1>("trestbps");
  features.insert(features.end(), tmp.begin(), tmp.end());
  tmp = doc.GetColumn<real1>("chol");
  features.insert(features.end(), tmp.begin(), tmp.end());
  tmp = doc.GetColumn<real1>("fbs");
  features.insert(features.end(), tmp.begin(), tmp.end());
  tmp = doc.GetColumn<real1>("restecg");
  features.insert(features.end(), tmp.begin(), tmp.end());
  tmp = doc.GetColumn<real1>("thalach");
  features.insert(features.end(), tmp.begin(), tmp.end());
  tmp = doc.GetColumn<real1>("exang");
  features.insert(features.end(), tmp.begin(), tmp.end());
  tmp = doc.GetColumn<real1>("oldpeak");
  features.insert(features.end(), tmp.begin(), tmp.end());
  tmp = doc.GetColumn<real1>("slope");
  features.insert(features.end(), tmp.begin(), tmp.end());
  tmp = doc.GetColumn<real1>("ca");
  features.insert(features.end(), tmp.begin(), tmp.end());
  tmp = doc.GetColumn<real1>("thal");
  features.insert(features.end(), tmp.begin(), tmp.end());

  std::vector<real1> target = doc.GetColumn<real1>("target");

  TensorPtr x = std::make_shared<Tensor>(features, std::vector<tcapint>{row_count, col_count}, std::vector<tcapint>{1, row_count});
  TensorPtr y = std::make_shared<Tensor>(target, std::vector<tcapint>{row_count, 1}, std::vector<tcapint>{1, 0U});

  const std::vector<ModulePtr> mv = {
      std::make_shared<Linear>(col_count, col_count << 1U), std::make_shared<Tanh>(),
      std::make_shared<Linear>(col_count << 1U, 1), std::make_shared<Sigmoid>()};

  Sequential model(mv);

  std::vector<ParameterPtr> params = model.parameters();

  Adam opt(R(0.001));
  opt.register_parameters(params);

  size_t epoch = 1;
  real1 loss_r = ONE_R1;

  while ((epoch <= 5000) && (loss_r > 0.01)) {
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
  std::ofstream o("heart_attack.qml");
  model.save(o);
  o.close();
}
