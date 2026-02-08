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

#include "modules/module.hpp"
#include "qrack/qneuron.hpp"
#include "tensors/tensor.hpp"

namespace Weed {
struct QrackNeuron : public Module {
  Qrack::QNeuron neuron;
  ParameterPtr angles;
  real1 *data;
  real1 denom;

  QrackNeuron(Qrack::QNeuron &qn);
  QrackNeuron(Qrack::QNeuron &qn, const std::vector<real1> &init_angles);

  TensorPtr forward() override;
};
typedef std::shared_ptr<QrackNeuron> QrackNeuronPtr;
} // namespace Weed
