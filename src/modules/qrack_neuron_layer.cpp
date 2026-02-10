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

#include "modules/qrack_neuron_layer.hpp"
#include "autograd/node.hpp"
#include "storage/cpu_real_storage.hpp"
#include "tensors/real_tensor.hpp"

#include <cstddef>
#include <functional>
#include <vector>

namespace Weed {
// Calls `fn(indices)` for each k-combination of {0..n-1}
inline void for_each_combination(
    size_t n, size_t k,
    const std::function<void(const std::vector<bitLenInt> &)> &fn) {
  if (k > n || k == 0)
    return;

  std::vector<bitLenInt> idx(k);
  for (size_t i = 0; i < k; ++i) {
    idx[i] = i;
  }

  while (true) {
    fn(idx);

    // Generate next combination
    size_t i = k;
    while (i-- > 0) {
      if (idx[i] != i + n - k) {
        break;
      }
    }
    if (i == SIZE_MAX) {
      return;
    }

    ++idx[i];
    for (size_t j = i + 1; j < k; ++j) {
      idx[j] = idx[j - 1] + 1;
    }
  }
}

std::function<void(Qrack::QInterfacePtr)>
QrackNeuronLayer::choose_quantum_fn(QuantumFunctionType fn) {
  switch (fn) {
  case NONE_QFN:
    return [](Qrack::QInterfacePtr s) {};
  case BELL_GHZ_QFN:
    return [&](Qrack::QInterfacePtr s) {
      s->H(input_indices[0U]);
      for (size_t i = 1U; i < input_indices.size(); ++i) {
        s->CNOT(input_indices[i - 1U], input_indices[i]);
      }
    };
  case ALT_BELL_GHZ_QFN:
    return [&](Qrack::QInterfacePtr s) {
      s->H(input_indices[0U]);
      for (size_t i = 1U; i < input_indices.size(); ++i) {
        s->AntiCNOT(input_indices[i - 1U], input_indices[i]);
      }
    };
  case CUSTOM_QFN:
  default:
    throw std::invalid_argument("Can't recognize QuantumFunctionType in "
                                "QrackNeuronLayer::choose_quantum_fn!");
  }
}

QrackNeuronLayer::QrackNeuronLayer(
    const size_t &input_q, const size_t &output_q, const size_t &hidden_q,
    const size_t &lowest_combo, const size_t &highest_combo,
    const Qrack::QNeuronActivationFn &activation,
    const QuantumFunctionType pre_fn, const QuantumFunctionType post_fn,
    const std::function<void(Qrack::QInterfacePtr)> &pre_init,
    const std::function<void(Qrack::QInterfacePtr)> &post_init,
    const real1_f &nw, const bool &md, const bool &sd, const bool &sh,
    const bool &bdt, const bool &pg, const bool &tn, const bool &hy,
    const bool &oc, const bool &hp, const bool &sp)
    : Module(QRACK_NEURON_LAYER_T), lowest_cmb(lowest_combo),
      highest_cmb(highest_combo), activation_fn(activation), pre_qfn(pre_fn),
      post_qfn(post_fn), input_indices(input_q), hidden_indices(hidden_q),
      output_indices(output_q), requires_grad(true) {

  if (pre_init && (pre_fn != CUSTOM_QFN)) {
    throw std::invalid_argument("Cannot specify a custom QrackNeuronLayer "
                                "pre_init without pre_fn = CUSTOM_QFN!");
  }
  if (post_init && (post_fn != CUSTOM_QFN)) {
    throw std::invalid_argument("Cannot specify a custom QrackNeuronLayer "
                                "pre_init without pre_fn = CUSTOM_QFN!");
  }

  const bitLenInt num_qubits = input_q + output_q + hidden_q;
  prototype = Qrack::CreateArrangedLayersFull(
      nw, md, sd, sh, bdt, pg, tn, hy, oc, num_qubits, Qrack::ZERO_BCI, nullptr,
      Qrack::CMPLX_DEFAULT_ARG, false, true, hp, sp);
  for (bitLenInt i = 0U; i < input_q; ++i) {
    input_indices[i] = i;
  }
  for (bitLenInt i = 0U; i < hidden_q; ++i) {
    hidden_indices[i] = i + input_q;
  }
  for (bitLenInt i = 0U; i < output_q; ++i) {
    output_indices[i] = i + input_q + hidden_q;
  }

  for (const auto &output_id : output_indices) {
    for (size_t k = lowest_combo; k < (highest_combo + 1U); ++k) {
      for_each_combination(
          input_q, k, [&](const std::vector<bitLenInt> &combo) {
            Qrack::QNeuronPtr qn =
                std::make_shared<Qrack::QNeuron>(prototype, combo, output_id);
            neurons.push_back(std::make_shared<QrackNeuron>(qn));
          });
    }
  }

  // Prepare hidden predictors
  for (const auto &hidden_id : hidden_indices) {
    prototype->H(hidden_id);
  }
  // Prepare a maximally uncertain output state.
  for (const auto &output_id : output_indices) {
    prototype->H(output_id);
  }

  update_param_vector();

  if (pre_init) {
    pre_init(prototype);
  } else {
    choose_quantum_fn(pre_qfn)(prototype);
  }

  post_init_fn = post_init ? post_init : choose_quantum_fn(post_qfn);
}

TensorPtr QrackNeuronLayer::forward(const TensorPtr x) {
  if (x->storage->dtype != DType::REAL) {
    throw std::invalid_argument(
        "QrackNeuronLayer::forward(x) argument must be real-number!");
  }

  const real1 init_phi = real1(asin(real1_f(0.5f)));

  const size_t B = x->shape[0];
  TensorPtr out = Tensor::zeros(
      std::vector<tcapint>{(tcapint)B, (tcapint)(output_indices.size())},
      requires_grad || x->requires_grad, true, DType::REAL, DeviceTag::CPU);
  TensorPtr in = std::make_shared<Tensor>(*(x.get()));

  in->storage = in->storage->cpu();

  const CpuRealStorage *pi =
      static_cast<const CpuRealStorage *>(in->storage.get());
  CpuRealStorage *po = static_cast<CpuRealStorage *>(out->storage.get());

  for (size_t b = 0U; b < B; ++b) {
    auto sim = prototype->Clone();

    // Load classical inputs
    for (size_t i = 0; i < input_indices.size(); ++i) {
      const real1 v =
          (*pi)[in->offset + b * in->stride[0U] + i * in->stride[1U]];
      sim->RY(PI_R1 * v, input_indices[i]);
    }

    post_init_fn(sim);

    for (size_t o = 0U; o < output_indices.size(); ++o) {
      real1 phi = init_phi;
      for (auto &neuron : neurons) {
        if (neuron->neuron->GetOutputIndex() == output_indices[o]) {
          TensorPtr d = neuron->forward(sim);
          RealTensor r = *static_cast<RealTensor *>(d.get());
          phi += r[0U];
        }
      }

      const real1 p = real1(std::max(std::sin(phi), real1_f(0)));
      po->write(out->offset + b * out->stride[0U] + o * out->stride[1U], p);
    }
  }

  return out;
}

void QrackNeuronLayer::save(std::ostream &os) const {
  if ((pre_qfn == CUSTOM_QFN) || (post_qfn == CUSTOM_QFN)) {
    throw std::domain_error("Can't serialize QrackNeuronLayer with custom "
                            "pre-or-post-initialization functions!");
  }

  Module::save(os);

  Serializer::write_tcapint(os, (tcapint)(input_indices.size()));
  Serializer::write_tcapint(os, (tcapint)(output_indices.size()));
  Serializer::write_tcapint(os, (tcapint)(hidden_indices.size()));
  Serializer::write_tcapint(os, (tcapint)lowest_cmb);
  Serializer::write_tcapint(os, (tcapint)highest_cmb);
  Serializer::write_qneuron_activation_fn(os, activation_fn);
  Serializer::write_quantum_fn(os, pre_qfn);
  Serializer::write_quantum_fn(os, post_qfn);

  for (size_t i = 0U; i < neurons.size(); ++i) {
    neurons[i]->angles->save(os);
  }
}
} // namespace Weed
