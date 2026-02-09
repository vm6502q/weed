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

#pragma once

#if !QRACK_AVAILABLE
#error Qrack files were included without Qrack available.
#endif

#include "modules/module.hpp"
#include "qrack/qneuron.hpp"
#include "tensors/tensor.hpp"

namespace Weed {
/**
 * Quantum perceptron model (which cannot be automatically serialized due to
 * reliance on auxiliary quantum simulator)
 */
struct QrackNeuron : public Module {
  Qrack::QNeuronPtr neuron;
  Qrack::QNeuronActivationFn activation_fn;
  ParameterPtr angles;
  real1 *data;

  QrackNeuron() : Module(QRACK_NEURON_T) {}
  QrackNeuron(Qrack::QNeuronPtr qn,
              const Qrack::QNeuronActivationFn &activation =
                  Qrack::QNeuronActivationFn::Sigmoid,
              const bool &init_rand = true);
  QrackNeuron(Qrack::QNeuronPtr qn, const std::vector<real1> &init_angles,
              const Qrack::QNeuronActivationFn &activation =
                  Qrack::QNeuronActivationFn::Sigmoid);

  std::vector<ParameterPtr> parameters() override {
    return std::vector<ParameterPtr>{angles};
  }

  void save(std::ostream &) const override {
    throw std::domain_error("Can't serialize QrackNeuron or quantum objects!");
  }

  TensorPtr forward(TensorPtr x) override {
    throw std::domain_error("QrackNeuron acts forward() on Qrack::QInterface "
                            "simulators, not Weed::Tensor!");
  }
  TensorPtr forward(Qrack::QInterfacePtr q, const std::vector<bitLenInt> &c,
                    const bitLenInt &t) override {
    if (c.size() != neuron->GetInputCount()) {
      throw std::invalid_argument(
          "Input size mismatch in QrackNeuron::forward!");
    }
    neuron->SetIndices(c, t);

    return forward(q);
  }
  TensorPtr forward(Qrack::QInterfacePtr q) override {
    neuron->SetSimulator(q);

    return forward();
  }
  TensorPtr forward() override;
};
typedef std::shared_ptr<QrackNeuron> QrackNeuronPtr;
} // namespace Weed
