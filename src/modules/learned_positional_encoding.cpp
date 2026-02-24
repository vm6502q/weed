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

#include "modules/learned_positional_encoding.hpp"
#include "common/serializer.hpp"

#include <random>

namespace Weed {

LearnedPositionalEncoding::LearnedPositionalEncoding(const tcapint &max_len_,
                                                     const tcapint &d_model_,
                                                     const DeviceTag &dtag)
    : Module(LEARNED_POSITIONAL_ENCODING_T), max_len(max_len_),
      d_model(d_model_) {

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<real1_s> dis(real1_s(0), real1_s(0.01));

  // initialize small random values
  std::vector<real1> init(max_len * d_model);
  for (auto &v : init) {
    v = (real1)dis(gen);
  }

  pos_encoding = std::make_shared<Parameter>(
      init, std::vector<tcapint>{1U, max_len, d_model}, dtag);
}

TensorPtr LearnedPositionalEncoding::forward(const TensorPtr x) {
  // x: (B, T, d_model)
  const auto &sh = x->shape;
  const symint T = sh[1];

  // Slice to actual sequence length if needed
  TensorPtr pos = Tensor::slice(pos_encoding, 1, 0, T);

  return x + pos;
}

void LearnedPositionalEncoding::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, max_len);
  Serializer::write_tcapint(os, d_model);
  pos_encoding->save(os);
}

} // namespace Weed
