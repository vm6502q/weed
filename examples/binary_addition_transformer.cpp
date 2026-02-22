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
#include "autograd/bci_loss.hpp"
#include "autograd/zero_grad.hpp"
#include "modules/embedding.hpp"
#include "modules/linear.hpp"
#include "modules/positional_encoding.hpp"
#include "modules/sequential.hpp"
#include "modules/sigmoid.hpp"
#include "modules/tanh.hpp"
#include "modules/transformer_encoder_layer.hpp"
#include "tensors/symbol_tensor.hpp"

#define GET_REAL(ptr) static_cast<RealScalar *>((ptr).get())->get_item()

#define R(v) real1(v)
#define C(v) complex(v)

using namespace Weed;

struct BinaryAdditionSample {
  std::vector<symint> input_tokens; // token ids
  std::vector<real1> target_bits;   // 0/1 labels
};

BinaryAdditionSample generate_samples(int bit_width, int samples) {
  BinaryAdditionSample batch;

  int max_val = 1 << bit_width;

  for (int j = 0; j < samples; ++j) {
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

    batch.input_tokens.insert(batch.input_tokens.end(),
                              sample.input_tokens.begin(),
                              sample.input_tokens.end());
    batch.target_bits.insert(batch.target_bits.end(),
                             sample.target_bits.begin(),
                             sample.target_bits.end());
  }

  return batch;
}

int main() {
  const tcapint bit_width = 4;
  const tcapint seq_len = bit_width * 2 + 2; // a + b =
  const tcapint target_len = bit_width + 1;  // sum bits
  const tcapint vocab_size = 5;

  const tcapint d_model = 32;
  const tcapint d_ff = 64;
  const tcapint num_heads = 4;

  const int epochs = 2000;
  const int batch_size = 32;

  // ---- Model ----
  Sequential model({
    std::make_shared<Embedding>(vocab_size, d_model),
    std::make_shared<PositionalEncoding>(seq_len, d_model),
    std::make_shared<TransformerEncoderLayer>(d_model, num_heads, d_ff),
    std::make_shared<TransformerEncoderLayer>(d_model, num_heads, d_ff),
    std::make_shared<Linear>(d_model, 1),
    std::make_shared<Tanh>(),
    std::make_shared<Linear>(1, 1),
    std::make_shared<Sigmoid>()}
  );

  Adam optimizer(R(0.01));
  optimizer.register_parameters(model.parameters());

  // ---- Training ----
  for (int epoch = 0; epoch < epochs; ++epoch) {
    auto sample = generate_samples(bit_width, batch_size);

    auto input = std::make_shared<SymbolTensor>(
        sample.input_tokens, std::vector<tcapint>{batch_size, seq_len},
        std::vector<tcapint>{1, batch_size});

    auto logits = model.forward(input);

    // We take only last (target_len) positions
    auto predicted =
        Tensor::slice(logits, 1, seq_len - target_len, target_len)->squeeze(2);

    auto target = std::make_shared<Tensor>(
        sample.target_bits, std::vector<tcapint>{batch_size, target_len},
        std::vector<tcapint>{1, batch_size});

    auto loss = bci_loss(predicted, target);

    Tensor::backward(loss);

    adam_step(optimizer, model.parameters());
    model.train();
    zero_grad(model.parameters());

    if (epoch % 100 == 0) {
      const real1_f loss_r = GET_REAL(loss);
      std::cout << "Epoch " << epoch << ", Loss: " << loss_r << std::endl;
    }
  }

  return 0;
}
