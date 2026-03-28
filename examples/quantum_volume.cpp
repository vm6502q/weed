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
#include "modules/sigmoid.hpp"
#include "modules/tanh.hpp"
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
  const int p = n << 1;
  const int s = p << 2;

  std::vector<bitLenInt> allBits(n);
  std::iota(allBits.begin(), allBits.end(), 0U);

  const std::vector<ModulePtr> mv = {
      std::make_shared<Linear>(p, p, true, false), std::make_shared<Sigmoid>(),
      std::make_shared<QrackNeuronLayer>(p, n, 0, n, n, BELL_GHZ_QFN),
      std::make_shared<Linear>(n, n, true, false)};

  Sequential model(mv);

  std::vector<ParameterPtr> params = model.parameters();

  Adam opt(R(0.001));
  opt.register_parameters(params);

  size_t epoch = 1;
  real1 loss_r = ONE_R1;

  while ((epoch <= 1000) && (loss_r > 0.001)) {
    std::vector<real1> a_th;
    std::vector<real1> a_ph;
    std::vector<real1> y_vec;

    for (int b = 0; b < s; ++b) {
      Qrack::QInterfacePtr qReg = Qrack::CreateQuantumInterface(
          Qrack::QINTERFACE_TENSOR_NETWORK, n, Qrack::ZERO_BCI);

      // Two parallel running products, one per X/Z pair
      std::vector<real1_f> det_th(n, 0.0);
      std::vector<real1_f> det_ph(n, 0.0);

      // Per-layer sin values, kept separate
      std::vector<real1_f> s_th(n), s_ph(n);
      std::vector<real1_f> th_last(n, 0.0);

      for (int d = 0; d < depth; ++d) {
        // Single-qubit layer
        for (bitLenInt i = 0U; i < n; ++i) {
          const real1_f th = (2 * qReg->Rand() - 1) * PI_R1;
          const real1_f ph = (2 * qReg->Rand() - 1) * PI_R1;
          qReg->AI(i, th, ph);

          th_last[i] = th;
          s_th[i] = th;
          s_ph[i] = ph;

          // Accumulate own contribution
          det_th[i] = fix_range(det_th[i] + th);
          det_ph[i] = fix_range(det_ph[i] + ph);
        }

        // Two-qubit layer
        std::vector<bitLenInt> unusedBits(allBits);
        while (unusedBits.size() > 1U) {
          const bitLenInt b1 = pickRandomBit(qReg->Rand(), &unusedBits);
          const bitLenInt b2 = pickRandomBit(qReg->Rand(), &unusedBits);
          qReg->CNOT(b1, b2);

          // Forward: propagate bit-flip state onto target
          det_ph[b2] = fix_range(det_ph[b2] + s_ph[b1]);

          // Reverse kickback: propagate phase-flip state onto contro
          det_th[b1] = fix_range(det_th[b1] + s_th[b2]);
        }

        // Unpaired qubit — already accumulated in single-qubit pass
        // nothing extra needed
      }

      a_th.insert(a_th.end(), det_th.begin(), det_th.end());
      a_ph.insert(a_ph.end(), det_ph.begin(), det_ph.end());

      for (bitLenInt i = 0U; i < n; ++i) {
        const real1_f prb = qReg->Prob(i);
        y_vec.push_back(std::log(prb / (1 - prb)));
      }
    }

    std::vector<real1> x_vec;
    x_vec.insert(x_vec.end(), a_th.begin(), a_th.end());
    x_vec.insert(x_vec.end(), a_ph.begin(), a_ph.end());

    TensorPtr x = std::make_shared<Tensor>(x_vec, std::vector<tcapint>{s, p});
    TensorPtr y = std::make_shared<Tensor>(y_vec, std::vector<tcapint>{s, n});

    TensorPtr y_pred = model.forward(x);
    TensorPtr loss = mse_loss(y_pred, y);

    Tensor::backward(loss);
    adam_step(opt, params);

    loss_r = GET_REAL(loss);
    if ((epoch % 100) == 0) {
      std::cout << "Epoch " << epoch << ", Loss: " << loss_r << std::endl;
    }

    zero_grad(params);
    ++epoch;
  }

  --epoch;
  if (epoch % 100) {
    std::cout << "Epoch " << epoch << ", Loss: " << loss_r << std::endl;
  }

  std::ofstream o("quantum_volume.qml");
  model.save(o);
  o.close();

  return 0;
}
