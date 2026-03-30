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

#include "modules/transformer_encoder_layer.hpp"
#include "common/serializer.hpp"

#include "modules/gelu.hpp"
#include "modules/relu.hpp"
#include "modules/sigmoid.hpp"
#include "modules/swiglu.hpp"
#include "modules/tanh.hpp"

namespace Weed {
TransformerEncoderLayer::TransformerEncoderLayer(
    const tcapint &d_model_, const tcapint &num_heads_, const tcapint &d_ff_,
    const DeviceTag &dtag, const ActivationFunctionType &afn,
    const int64_t &did)
    : Module(TRANSFORMER_ENCODER_LAYER_T), d_model(d_model_), d_ff(d_ff_),
      num_heads(num_heads_),
      self_attn(std::make_shared<MultiHeadAttention>(
          d_model_, num_heads_, num_heads_, 0U, dtag, nullptr, ZERO_R1, did)),
      ff1(std::make_shared<Linear>(d_model_, d_ff_, true, true, DType::REAL,
                                   dtag, did)),
      ff2(std::make_shared<Linear>(d_ff_, d_model_, true, true, DType::REAL,
                                   dtag, did)),
      norm1(std::make_shared<LayerNorm>(d_model_, dtag, FP_NORM_EPSILON, did)),
      norm2(std::make_shared<LayerNorm>(d_model_, dtag, FP_NORM_EPSILON, did)) {
  switch (afn) {
  case SIGMOID_FN:
    activation = std::make_shared<Sigmoid>();
    break;
  case TANH_FN:
    activation = std::make_shared<Tanh>();
    break;
  case RELU_FN:
    activation = std::make_shared<ReLU>();
    break;
  case SWIGLU_FN:
    activation = std::make_shared<SwiGLU>();
    break;
  case GELU_FN:
  default:
    activation = std::make_shared<GeLU>();
  }

  param_vector = self_attn->parameters();
  auto add = [&](const std::vector<ParameterPtr> &q) {
    param_vector.insert(param_vector.end(), q.begin(), q.end());
  };
  add(ff1->parameters());
  add(ff2->parameters());
  add(norm1->parameters());
  add(norm2->parameters());
}
TensorPtr TransformerEncoderLayer::forward(const TensorPtr x_) {
  size_t sz = x_->storage->size;
  std::vector<BaseTensorPtr> t_vec{x_};
  for (const auto &p : param_vector) {
    t_vec.push_back(p);
    sz += p->storage->size;
  }
  const DeviceTag dtag = Tensor::get_dtag_by_presidence(t_vec);
  TensorPtr x = x_->cast(dtag);
  const bool isGpu = (dtag == GPU);
  const bool isRevert =
      !_weed_telescope_transformers &&
      ((24 * sz * sizeof(complex)) > _weed_max_ocl_b);

  if (isGpu) {
    norm1->migrate_gpu();
  }
  TensorPtr x1 = norm1->forward(x);
  if (isGpu) {
    if (isRevert) {
      norm1->migrate_cpu();
    }
    self_attn->migrate_gpu();
  }
  x1 = self_attn->forward(x1);
  if (isGpu && isRevert) {
    self_attn->migrate_cpu();
  }
  x1 = x + x1;

  if (isGpu) {
    norm2->migrate_gpu();
  }
  TensorPtr ff = norm2->forward(x1);
  if (isGpu) {
    if (isRevert) {
      norm2->migrate_cpu();
    }
    ff1->migrate_gpu();
  }
  ff = ff1->forward(ff);
  if (isGpu) {
    if (isRevert) {
      ff1->migrate_cpu();
    }
    activation->migrate_gpu();
  }
  ff = activation->forward(ff);
  if (isGpu) {
    if (isRevert) {
      activation->migrate_cpu();
    }
    ff2->migrate_gpu();
  }
  ff = ff2->forward(ff);
  if (isGpu && isRevert) {
    ff2->migrate_cpu();
  }
  ff = x1 + ff;

  x1 = nullptr;

  return ff;
}
void TransformerEncoderLayer::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, d_model);
  Serializer::write_tcapint(os, d_ff);
  Serializer::write_tcapint(os, num_heads);
  self_attn->save(os);
  ff1->save(os);
  ff2->save(os);
  norm1->save(os);
  norm2->save(os);
  activation->save(os);
}
} // namespace Weed
