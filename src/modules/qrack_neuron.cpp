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
QrackNeuron::QrackNeuron(Qrack::QNeuron &qn)
    : Module(QRACK_NEURON), neuron(qn) {
  angles = std::make_shared<Parameter>(
      std::vector<tcapint>{(tcapint)(qn.GetInputPower())},
      std::vector<tcapint>{1U}, DType::REAL, DeviceTag::CPU, false);
  angles->storage->FillZeros();
  data = static_cast<CpuRealStorage *>(angles->storage.get())->data.get();
}

QrackNeuron::QrackNeuron(Qrack::QNeuron &qn,
                         const std::vector<real1> &init_angles)
    : Module(QRACK_NEURON), neuron(qn) {
  angles = std::make_shared<Parameter>(
      init_angles, std::vector<tcapint>{(tcapint)(qn.GetInputPower())},
      std::vector<tcapint>{1U}, DeviceTag::CPU);
  data = static_cast<CpuRealStorage *>(angles->storage.get())->data.get();
}

TensorPtr QrackNeuron::forward() {
  // x is angles tensor

  // Save simulator state
  const real1 pre_prob = neuron.GetSimulator()->Prob(neuron.GetOutputIndex());
  const real1 post_prob = neuron.Predict(data, true, false);
  const real1 delta = std::asin(post_prob) - std::asin(pre_prob);
  denom = std::max(std::sqrt(std::max(ONE_R1 - post_prob * post_prob, ZERO_R1)),
                   FP_NORM_EPSILON);

  TensorPtr out = std::make_shared<Tensor>(delta, angles->requires_grad);

  if (angles->requires_grad) {
    out->make_gradient();
    out->grad_node =
        std::make_shared<Node>(std::vector<TensorPtr>{angles}, [this, out]() {
          const DeviceTag dtag =
              Tensor::get_dtag_by_presidence({angles->grad, out->grad});
          TensorPtr dx = angles->grad->cast(dtag);
          TensorPtr dout = out->grad->cast(dtag);
          dx->storage = dx->storage->cpu();
          dout->storage = dout->storage->cpu();

          RealTensor dxo = *static_cast<RealTensor *>(dx.get());
          RealTensor doo = *static_cast<RealTensor *>(dout.get());

          neuron.Unpredict(data, true);

          const real1 upstream = doo[0U] / denom;

          const tcapint max_lcv = angles->get_broadcast_size();
          for (size_t i = 0U; i < max_lcv; ++i) {
            real1 theta = data[i];

            // +π/2
            data[i] = theta + SineShift;
            real1 p_plus = neuron.Predict(data, true, false);
            neuron.Unpredict(data, true);

            // -π/2
            data[i] = theta - SineShift;
            real1 p_minus = neuron.Predict(data, true, false);
            neuron.Unpredict(data, true);

            real1 grad = (p_plus - p_minus) / 2;
            dxo.add(i, grad * upstream);

            data[i] = theta;
          }

          angles->grad = dx;
        });
  }

  return out;
}

} // namespace Weed
