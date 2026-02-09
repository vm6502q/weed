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
#include "modules/qrack_neuron.hpp"

namespace Weed {
struct QrackNeuronLayer : public Module {
  Qrack::QInterfacePtr prototype;
  std::vector<bitLenInt> input_indices;
  std::vector<bitLenInt> hidden_indices;
  std::vector<bitLenInt> output_indices;
  Qrack::QNeuronActivationFn activation_fn;
  std::function<void(Qrack::QInterfacePtr)> post_init_fn;
  std::vector<QrackNeuronPtr> neurons;
  std::vector<ParameterPtr> param_vector;
  bool requires_grad;

  QrackNeuronLayer(
      const size_t &input_q, const size_t &output_q, const size_t &hidden_q,
      const size_t &lowest_combo, const size_t &highest_combo,
      const Qrack::QNeuronActivationFn &activation =
          Qrack::QNeuronActivationFn::Sigmoid,
      const std::function<void(Qrack::QInterfacePtr)> &post_init =
          [](Qrack::QInterfacePtr q) {},
      const real1_f &nw = ZERO_R1, const bool &md = false,
      const bool &sd = true, const bool &sh = true, const bool &bdt = false,
      const bool &pg = true, const bool &tn = true, const bool &hy = true,
      const bool &oc = true, const bool &hp = false, const bool &sp = false);

  std::vector<ParameterPtr> parameters() override { return param_vector; }

  void train() override {
    for (const QrackNeuronPtr &n : neurons) {
      n->train();
    }
    requires_grad = true;
  }
  void eval() override {
    for (const QrackNeuronPtr &n : neurons) {
      n->eval();
    }
    requires_grad = false;
  }

  void save(std::ostream &) const override {
    throw std::domain_error("Can't serialize QrackNeuron or quantum objects!");
  }

  TensorPtr forward(const TensorPtr x) override;
};
typedef std::shared_ptr<QrackNeuronLayer> QrackNeuronLayerPtr;
} // namespace Weed
