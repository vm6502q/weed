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

#include "modules/qrack_neuron.hpp"
#include "autograd/node.hpp"
#include "storage/cpu_real_storage.hpp"
#include "tensors/real_tensor.hpp"

namespace Weed {
QrackNeuron::QrackNeuron(Qrack::QNeuronPtr qn,
                         const Qrack::QNeuronActivationFn &activation,
                         const bool &init_rand)
    : Module(QRACK_NEURON_T), neuron(qn), activation_fn(activation) {
  const size_t sz = qn->GetInputPower();
  if (init_rand) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real1_s> dis(-PI_R1, PI_R1);
    std::vector<real1> init;
    init.reserve(sz);
    for (size_t n = 0; n < sz; ++n) {
      init.push_back((real1)dis(gen));
    }

    angles =
        std::make_shared<Parameter>(init, std::vector<tcapint>{(tcapint)sz},
                                    std::vector<tcapint>{1U}, DeviceTag::CPU);
  } else {
    angles = std::make_shared<Parameter>(std::vector<tcapint>{(tcapint)sz},
                                         std::vector<tcapint>{1U}, false,
                                         DType::REAL, DeviceTag::CPU, -1);
    angles->storage->FillZeros();
  }
  data = static_cast<CpuRealStorage *>(angles->storage.get())->data.get();
}

QrackNeuron::QrackNeuron(Qrack::QNeuronPtr qn,
                         const std::vector<real1> &init_angles,
                         const Qrack::QNeuronActivationFn &activation)
    : Module(QRACK_NEURON_T), neuron(qn), activation_fn(activation) {
  angles = std::make_shared<Parameter>(
      init_angles, std::vector<tcapint>{(tcapint)(qn->GetInputPower())},
      std::vector<tcapint>{1U}, DeviceTag::CPU);
  data = static_cast<CpuRealStorage *>(angles->storage.get())->data.get();
}

TensorPtr QrackNeuron::forward() {
  // x is angles tensor

  // Save simulator state
  const real1_f pre_prob =
      neuron->GetSimulator()->Prob(neuron->GetOutputIndex());
  const real1_f post_prob = neuron->Predict(data, true, false, activation_fn);
  const real1_f delta = std::asin(post_prob) - std::asin(pre_prob);
  const real1_f denom =
      std::max(std::sqrt(std::max(ONE_R1 - post_prob * post_prob, real1_f(0))),
               real1_f(FP_NORM_EPSILON));

  TensorPtr out = std::make_shared<Tensor>(real1(delta), angles->requires_grad);

  if (angles->requires_grad) {
    out->make_gradient();
    out->grad_node = std::make_shared<Node>(
        std::vector<TensorPtr>{angles}, [this, denom, out]() {
          TensorPtr dx = angles->grad;
          TensorPtr dout = out->grad;

          dx->storage = dx->storage->cpu();
          dout->storage = dout->storage->cpu();

          RealTensor dxo = *static_cast<RealTensor *>(dx.get());
          const RealTensor doo = *static_cast<const RealTensor *>(dout.get());

          neuron->Unpredict(data, true, activation_fn);

          const real1_f upstream = doo[0U] / denom;

          const tcapint max_lcv = angles->get_broadcast_size();
          for (size_t i = 0U; i < max_lcv; ++i) {
            const real1 theta = data[i];

            // +π/2
            data[i] = theta + SineShift;
            const real1_f p_plus =
                neuron->Predict(data, true, false, activation_fn);
            neuron->Unpredict(data, true, activation_fn);

            // -π/2
            data[i] = theta - SineShift;
            const real1_f p_minus =
                neuron->Predict(data, true, false, activation_fn);
            neuron->Unpredict(data, true, activation_fn);

            const real1_f grad = (p_plus - p_minus) / 2;
            dxo.add(i, real1(grad * upstream));

            data[i] = theta;
          }

          angles->grad = dx;
        });
  }

  return out;
}
} // namespace Weed
