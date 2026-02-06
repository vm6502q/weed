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
#include "common/serializer.hpp"
#include "storage/all_storage.hpp"

#include <random>

namespace Weed {
Linear::Linear(tcapint in_f, tcapint out_f, bool use_bias, DType dtype,
               DeviceTag device, int64_t device_id, bool init_rand)
    : Module(LINEAR_T), in_features(in_f), out_features(out_f) {

  const std::vector<tcapint> shape{in_f, out_f};
  const std::vector<tcapint> stride{1U, in_f};

  if (init_rand) {
    std::random_device rd;
    std::mt19937 gen(rd());

    real1_s lim = (real1_s)(0.5 / std::sqrt(in_f));
    std::uniform_real_distribution<real1_s> dis(-lim, lim);

    const size_t sz = in_f * out_f;
    if (dtype == DType::REAL) {
      std::vector<real1> init;
      init.reserve(sz);
      for (size_t n = 0; n < sz; ++n) {
        init.push_back((real1)dis(gen));
      }

      weight =
          std::make_shared<Parameter>(init, shape, stride, device, device_id);
    } else {
      std::uniform_real_distribution<real1_s> adis((real1_s)(-PI_R1),
                                                   (real1_s)PI_R1);
      std::vector<complex> init;
      init.reserve(sz);
      for (size_t n = 0; n < sz; ++n) {
        init.push_back(std::polar((real1)dis(gen), (real1)adis(gen)));
      }

      weight =
          std::make_shared<Parameter>(init, shape, stride, device, device_id);
    }
  } else {
    weight =
        std::make_shared<Parameter>(shape, stride, dtype, device, device_id);
    weight->storage->FillZeros();
  }

  if (use_bias) {
    bias = std::make_shared<Parameter>(std::vector<tcapint>{out_f},
                                       std::vector<tcapint>{1U}, dtype, device,
                                       device_id);
    bias->storage->FillZeros();
  } else {
    bias = nullptr;
  }
}

TensorPtr Linear::forward(const TensorPtr x) {
  // x: (B, in_features)
  // W: (in_features, out_features)
  // We want: x @ W → (B, out_features)

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
void Linear::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, in_features);
  Serializer::write_tcapint(os, out_features);
  weight->save(os);
  Serializer::write_bool(os, !!bias);
  if (bias) {
    bias->save(os);
  }
}
} // namespace Weed
