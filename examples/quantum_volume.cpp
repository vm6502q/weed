//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
//
// This example demonstrates the quantum volume random unitary circuit
// generation protocol. It also "hashes" a "determinant" from the random circuit
// generated. This was developed in collaboration between Dan Strano and
// (Anthropic) Claude.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#include "qrack/qfactory.hpp"

#include "autograd/adam.hpp"
#include "autograd/mse_loss.hpp"
#include "autograd/zero_grad.hpp"
#include "modules/linear.hpp"
#include "modules/qrack_neuron_layer.hpp"
#include "modules/sequential.hpp"
#include "tensors/real_scalar.hpp"

#include <cmath>
#include <fstream>
#include <numeric>
#include <set>
#include <vector>

#define GET_REAL(ptr) static_cast<RealScalar *>((ptr).get())->get_item()

#define R(v) real1(v)
#define C(v) complex(v)

using namespace Weed;

bitLenInt pickRandomBit(real1_f rand, std::vector<bitLenInt> *unusedBitsPtr) {
  bitLenInt bitRand = (bitLenInt)(unusedBitsPtr->size() * rand);
  if (bitRand >= unusedBitsPtr->size()) {
    bitRand = unusedBitsPtr->size() - 1U;
  }
  bitLenInt result = (*unusedBitsPtr)[bitRand];
  unusedBitsPtr->erase(unusedBitsPtr->begin() + bitRand);

  return result;
}

real1_f fix_range(real1_f theta) {
  while (theta <= -PI_R1) {
    theta += 2 * PI_R1;
  }
  while (theta > PI_R1) {
    theta -= 2 * PI_R1;
  }

  return theta;
}

int main() {
  const bitLenInt n = 2;
  const int depth = n; // QV uses square circuits
  const int p = 3 * n;

  const std::vector<ModulePtr> mv = {
      std::make_shared<Linear>(p, p),
      std::make_shared<QrackNeuronLayer>(p, n, 0, p, p, BELL_GHZ_QFN)};

  Sequential model(mv);

  std::vector<ParameterPtr> params = model.parameters();

  Adam opt(R(0.1));
  opt.register_parameters(params);

  size_t epoch = 1;
  real1 loss_r = ONE_R1;

  while ((epoch <= 2000) && (loss_r > 0.01)) {
    Qrack::QInterfacePtr qReg = Qrack::CreateQuantumInterface(
        Qrack::QINTERFACE_TENSOR_NETWORK, n, Qrack::ZERO_BCI);

    std::vector<bitLenInt> allBits(n);
    std::iota(allBits.begin(), allBits.end(), 0U);

    // Three parallel running products, one per U3 parameter
    std::vector<real1> det_theta(n, 0.0);
    std::vector<real1> det_phi(n, 0.0);
    std::vector<real1> det_lambda(n, 0.0);

    // Per-layer sin values, kept separate
    std::vector<real1_f> s_theta(n), s_phi(n), s_lambda(n);
    std::vector<real1_f> theta_last(n, 0.0);

    for (int d = 0; d < depth; ++d) {
      // Single-qubit layer
      for (bitLenInt i = 0U; i < n; ++i) {
        const real1_f theta = 2 * PI_R1 * qReg->Rand() - PI_R1;
        const real1_f phi = 2 * PI_R1 * qReg->Rand() - PI_R1;
        const real1_f lambda = 2 * PI_R1 * qReg->Rand() - PI_R1;
        qReg->U(i, theta, phi, lambda);

        theta_last[i] = theta;
        s_theta[i] = theta;
        s_phi[i] = phi;
        s_lambda[i] = lambda;

        // Accumulate own contribution
        det_theta[i] = fix_range(det_theta[i] + theta);
        det_phi[i] = fix_range(det_phi[i] + phi);
        det_lambda[i] = fix_range(det_lambda[i] + lambda);
      }

      // Two-qubit layer
      std::vector<bitLenInt> unusedBits(allBits);
      while (unusedBits.size() > 1U) {
        const bitLenInt b1 = pickRandomBit(qReg->Rand(), &unusedBits);
        const bitLenInt b2 = pickRandomBit(qReg->Rand(), &unusedBits);
        qReg->CNOT(b1, b2);

        // Forward: propagate each parameter of control into target
        det_theta[b2] = fix_range(det_theta[b2] + s_theta[b1]);
        det_phi[b2] = fix_range(det_phi[b2] + s_phi[b1]);
        det_lambda[b2] = fix_range(det_lambda[b2] + s_lambda[b1]);

        // Reverse kickback: cos(theta_b2/2) scales back into control
        // Applied to all three parameters of the control qubit
        const real1_f kickback = (PI_R1 + theta_last[b2]) / 2;
        det_theta[b1] = fix_range(det_theta[b1] + kickback);
        det_phi[b1] = fix_range(det_phi[b1] + kickback);
        det_lambda[b1] = fix_range(det_lambda[b1] + kickback);
      }

      // Unpaired qubit — already accumulated in single-qubit pass
      // nothing extra needed
    }

    std::vector<real1> x_vec;
    x_vec.insert(x_vec.end(), det_theta.begin(), det_theta.end());
    x_vec.insert(x_vec.end(), det_phi.begin(), det_phi.end());
    x_vec.insert(x_vec.end(), det_lambda.begin(), det_lambda.end());

    std::vector<real1> y_vec;
    for (bitLenInt i = 0U; i < n; ++i) {
      y_vec.push_back(qReg->Prob(i));
    }

    TensorPtr x = std::make_shared<Tensor>(x_vec, std::vector<tcapint>{1, p});
    TensorPtr y = std::make_shared<Tensor>(y_vec, std::vector<tcapint>{1, n});

    TensorPtr y_pred = model.forward(x);
    TensorPtr loss = mse_loss(y_pred, y);

    Tensor::backward(loss);
    adam_step(opt, params);

    loss_r = GET_REAL(loss);
    std::cout << "Epoch " << epoch << ", Loss: " << loss_r << std::endl;

    zero_grad(params);
    ++epoch;
  }

  std::ofstream o("quantum_volume.qml");
  model.save(o);
  o.close();

  return 0;
}
