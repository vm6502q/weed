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

#include "modules/linear.hpp"
#include "storage/all_storage.hpp"

#include <random>

namespace Weed {
Linear::Linear(vecCapIntGpu in_f, vecCapIntGpu out_f, bool use_bias,
               DType dtype, DeviceTag device, int64_t device_id)
    : in_features(in_f), out_features(out_f) {

  const std::vector<vecCapIntGpu> shape{in_f, out_f};
  const std::vector<vecCapIntGpu> stride{1, in_f};

  std::random_device rd;
  std::mt19937 gen(rd());

  const size_t sz = in_f * out_f;
  if (dtype == DType::REAL) {
    real1_f lim = (real1_f)(1.0 / std::sqrt(in_f));
    std::uniform_real_distribution<real1_f> dis(-lim, lim);
    std::vector<real1> init;
    init.reserve(sz);
    for (size_t n = 0; n < sz; ++n) {
      init.push_back((real1)dis(gen));
    }

    weight =
        std::make_shared<Parameter>(init, shape, stride, device, device_id);
  } else {
    real1_f lim = (real1_f)(1.0 / std::pow(in_f, 0.25));
    std::uniform_real_distribution<real1_f> dis(-lim, lim);
    std::vector<complex> init;
    init.reserve(sz);
    for (size_t n = 0; n < sz; ++n) {
      init.push_back(complex((real1)dis(gen), (real1)dis(gen)));
    }

    weight =
        std::make_shared<Parameter>(init, shape, stride, device, device_id);
  }

  if (use_bias) {
    bias = std::make_shared<Parameter>(std::vector<vecCapIntGpu>{out_f},
                                       std::vector<vecCapIntGpu>{1}, dtype,
                                       device, device_id);
    bias->storage->FillZeros();
  } else {
    bias = nullptr;
  }
}

TensorPtr Linear::forward(const TensorPtr x) {
  // x: (B, in_features)
  // W: (out_features, in_features)
  // We want: x @ W^T → (B, out_features)

  TensorPtr y = x >> weight;

  if (bias) {
    // bias shape: (out_features)
    // broadcast over batch dimension via stride
    y = y + bias;
  }

  return y;
}

std::vector<ParameterPtr> Linear::parameters() {
  if (bias) {
    return {weight, bias};
  }

  return {weight};
}
} // namespace Weed
