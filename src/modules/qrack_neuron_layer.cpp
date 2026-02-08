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

QrackNeuronLayer::QrackNeuronLayer(
    const size_t &input_q, const size_t &output_q, const size_t &hidden_q,
    const size_t &lowest_combo, const size_t &highest_combo,
    const Qrack::QNeuronActivationFn &activation,
    const std::function<void(Qrack::QInterfacePtr)> &post_init,
    const real1_f &nw, const bool &md, const bool &sd, const bool &sh,
    const bool &bdt, const bool &pg, const bool &tn, const bool &hy,
    const bool &oc, const bool &hp, const bool &sp)
    : Module(QRACK_NEURON_LAYER), input_indices(input_q),
      hidden_indices(hidden_q), output_indices(output_q),
      activation_fn(activation), post_init_fn(post_init) {
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
    for (size_t k = lowest_combo; k < (highest_combo + 1); ++k) {
      for_each_combination(
          input_q, k, [&](const std::vector<bitLenInt> &combo) {
            Qrack::QNeuron qn(prototype, combo, output_id);
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

  for (const auto &n : neurons) {
    const std::vector<ParameterPtr> p = n->parameters();
    param_vector.insert(param_vector.end(), p.begin(), p.end());
  }
}

TensorPtr QrackNeuronLayer::forward(const TensorPtr x) {
  WEED_CONST real1 init_phi = asin(ONE_R1 / 2);

  const size_t B = x->shape[0];
  TensorPtr out = Tensor::zeros(
      std::vector<tcapint>{(tcapint)B, (tcapint)(output_indices.size())},
      x->requires_grad, x->storage->dtype, DeviceTag::CPU);
  TensorPtr in = std::make_shared<Tensor>(*(x.get()));

  in->storage = in->storage->cpu();

  CpuRealStorage *pi = static_cast<CpuRealStorage *>(in->storage.get());
  CpuRealStorage *po = static_cast<CpuRealStorage *>(out->storage.get());

  for (size_t b = 0; b < B; ++b) {
    auto sim = prototype->Clone();

    // Load classical inputs
    for (size_t i = 0; i < input_indices.size(); ++i) {
      real1 v = (*pi)[in->offset + b * in->stride[0U] + i * in->stride[1U]];
      sim->RY(PI_R1 * v, input_indices[i]);
    }

    post_init_fn(sim);

    for (size_t o = 0; o < output_indices.size(); ++o) {
      real1 phi = init_phi;

      for (auto &neuron : neurons) {
        if (neuron->neuron.GetOutputIndex() == output_indices[o]) {
          TensorPtr d = neuron->forward(sim);
          RealTensor r = *static_cast<RealTensor *>(d.get());
          phi += r[0U];
        }
      }

      real1 p = std::max(std::sin(phi), real1(0));
      po->write(out->offset + b * out->stride[0U] + o * out->stride[1U], p);
    }
  }

  return out;
}
} // namespace Weed
