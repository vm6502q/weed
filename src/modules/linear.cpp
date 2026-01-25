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

#define SWITCH_DTYPE(val, strg, ctype, rtype)                                  \
  switch (dtype) {                                                             \
  case DType::COMPLEX:                                                         \
    static_cast<ctype *>(strg.get())->FillValue(ONE_R1 / 2);                   \
    break;                                                                     \
  case DType::REAL:                                                            \
  default:                                                                     \
    static_cast<rtype *>(strg.get())->FillValue(ONE_R1 / 2);                   \
  }

#define FILL_STORAGE(val, strg)                                                \
  switch (device) {                                                            \
  case DeviceTag::GPU:                                                         \
    SWITCH_DTYPE(val, strg, GpuComplexStorage, GpuRealStorage);                \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    SWITCH_DTYPE(val, strg, CpuComplexStorage, CpuRealStorage);                \
  }

namespace Weed {
Linear::Linear(vecCapIntGpu in_f, vecCapIntGpu out_f, bool use_bias,
               DType dtype, DeviceTag device, int64_t device_id)
    : in_features(in_f), out_features(out_f) {
  // weight is stored as W^T
  weight = std::make_shared<Parameter>(std::vector<vecCapIntGpu>{in_f, out_f},
                                       std::vector<vecCapIntGpu>{1, in_f},
                                       dtype, device, device_id);

  weight->storage->FillZeros();

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
