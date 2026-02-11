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

#include "enums/quantum_function_type.hpp"
#include "modules/module.hpp"
#include "modules/qrack_neuron.hpp"

namespace Weed {
/**
 * QNN layer based on quantum perceptron model (which cannot be automatically
 * serialized due to reliance on auxiliary quantum simulator and
 * post-initialization function pointer)
 */
struct QrackNeuronLayer : public Module {
  size_t lowest_cmb;
  size_t highest_cmb;
  QuantumFunctionType pre_qfn;
  QuantumFunctionType post_qfn;
  Qrack::QNeuronActivationFn activation_fn;

  Qrack::QInterfacePtr prototype;
  std::vector<bitLenInt> input_indices;
  std::vector<bitLenInt> hidden_indices;
  std::vector<bitLenInt> output_indices;
  std::function<void(Qrack::QInterfacePtr)> post_init_fn;
  std::vector<QrackNeuronPtr> neurons;
  std::vector<ParameterPtr> param_vector;
  bool requires_grad;

  tcapint qrack_config_mask;

  QrackNeuronLayer(
      const size_t &input_q, const size_t &output_q, const size_t &hidden_q,
      const size_t &lowest_combo, const size_t &highest_combo,
      const QuantumFunctionType pre_fn = BELL_GHZ_QFN,
      const QuantumFunctionType post_fn = NONE_QFN,
      const Qrack::QNeuronActivationFn &activation =
          Qrack::QNeuronActivationFn::Sigmoid,
      const std::function<void(Qrack::QInterfacePtr)> &pre_init = nullptr,
      const std::function<void(Qrack::QInterfacePtr)> &post_init = nullptr,
      const bool &md = false, const bool &sd = true, const bool &bdt = false,
      const bool &hp = false, const bool &sp = false);

  void update_param_vector() {
    param_vector.clear();
    for (const auto &n : neurons) {
      const std::vector<ParameterPtr> p = n->parameters();
      param_vector.insert(param_vector.end(), p.begin(), p.end());
    }
  }

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

  TensorPtr forward(const TensorPtr x) override;
  void save(std::ostream &) const override;

  std::function<void(Qrack::QInterfacePtr)>
  choose_quantum_fn(QuantumFunctionType fn);
};
typedef std::shared_ptr<QrackNeuronLayer> QrackNeuronLayerPtr;
} // namespace Weed
