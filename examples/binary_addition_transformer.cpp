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

#include "autograd/adam.hpp"
#include "autograd/bci_with_logits_loss.hpp"
#include "autograd/zero_grad.hpp"
#include "modules/embedding.hpp"
#include "modules/linear.hpp"
#include "modules/positional_encoding.hpp"
#include "modules/sequential.hpp"
#include "modules/transformer_encoder_layer.hpp"
#include "tensors/real_scalar.hpp"
#include "tensors/symbol_tensor.hpp"

#define GET_REAL(ptr) static_cast<RealScalar *>((ptr).get())->get_item()

#define R(v) real1(v)
#define C(v) complex(v)

#include "autograd/adam.hpp"
#include "autograd/bci_with_logits_loss.hpp"
#include "modules/embedding.hpp"
#include "modules/linear.hpp"
#include "modules/positional_encoding.hpp"
#include "modules/sequential.hpp"
#include "modules/transformer_encoder_layer.hpp"

using namespace Weed;

struct BinaryAdditionSample {
  std::vector<symint> input_tokens; // token ids
  std::vector<real1> target_bits;   // 0/1 labels
};

BinaryAdditionSample generate_sample(int bit_width) {
  int max_val = 1 << bit_width;

  int a = rand() % max_val;
  int b = rand() % max_val;
  int sum = a + b;

  BinaryAdditionSample sample;

  // Encode a (MSB first)
  for (int i = bit_width - 1; i >= 0; --i) {
    sample.input_tokens.push_back((a >> i) & 1);
  }

  sample.input_tokens.push_back(2); // '+'

  // Encode b
  for (int i = bit_width - 1; i >= 0; --i) {
    sample.input_tokens.push_back((b >> i) & 1);
  }

  sample.input_tokens.push_back(3); // '='

  // Encode result (bit_width + 1 bits)
  for (int i = bit_width; i >= 0; --i) {
    sample.target_bits.push_back((sum >> i) & 1);
  }

  return sample;
}

int main() {
  const int bit_width = 4;
  const int seq_len = bit_width * 2 + 2; // a + b =
  const int target_len = bit_width + 1;  // sum bits
  const int vocab_size = 5;

  const int d_model = 32;
  const int d_ff = 64;
  const int num_heads = 4;

  const int epochs = 2000;
  const int batch_size = 32;

  // ---- Model ----
  auto embedding = std::make_shared<Embedding>(vocab_size, d_model);
  auto pos_enc = std::make_shared<PositionalEncoding>(seq_len, d_model);

  auto encoder =
      std::make_shared<TransformerEncoderLayer>(d_model, num_heads, d_ff);

  auto output = std::make_shared<Linear>(d_model, 1);

  Sequential model({embedding, pos_enc, encoder, output});
  std::vector<ParameterPtr> params = model.parameters();

  Adam opt(R(0.001));
  opt.register_parameters(params);

  // ---- Training ----
  size_t epoch = 1;
  real1 total_loss = ONE_R1;
  while ((epoch <= epochs) && (total_loss > 0.01)) {
    total_loss = 0.0f;

    for (int b = 0; b < batch_size; ++b) {
      auto sample = generate_sample(bit_width);

      auto input = std::make_shared<SymbolTensor>(
          sample.input_tokens,
          std::vector<tcapint>{1U, (tcapint)sample.input_tokens.size()},
          std::vector<tcapint>{0U, 1U});

      auto logits = model.forward(input)->squeeze(2);

      // We take only last (target_len) positions
      auto predicted =
          Tensor::slice(logits, 1, seq_len - target_len, target_len);

      auto target =
          std::make_shared<Tensor>(
              sample.target_bits,
              std::vector<tcapint>{1U, (tcapint)sample.target_bits.size()},
              std::vector<tcapint>{0U, 1U})
              ->unsqueeze(0);

      auto loss = bci_with_logits_loss(predicted, target);
      total_loss += GET_REAL(loss);

      Tensor::backward(loss);
    }

    adam_step(opt, params);

    if ((epoch % 100) == 0U) {
      std::cout << "Epoch " << epoch << ", Loss: " << total_loss << std::endl;
    }

    zero_grad(params);
    ++epoch;
  }

  return 0;
}
