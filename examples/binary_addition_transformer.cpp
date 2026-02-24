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
#include "modules/learned_positional_encoding.hpp"
#include "modules/linear.hpp"
#include "modules/sequential.hpp"
#include "modules/transformer_encoder_layer.hpp"
#include "tensors/real_scalar.hpp"
#include "tensors/symbol_tensor.hpp"

#define GET_REAL(ptr) static_cast<RealScalar *>((ptr).get())->get_item()

#define R(v) real1(v)
#define C(v) complex(v)

using namespace Weed;

struct BinaryAdditionSample {
  std::vector<symint> input_tokens; // token ids
  std::vector<real1> target_bits;   // 0/1 labels
};

BinaryAdditionSample generate_samples(int bit_width) {
  std::vector<BinaryAdditionSample> svec;
  const int max_val = 1 << bit_width;
  const int max_lcv = 1 << (bit_width << 1U);
  for (int j = 0; j < max_lcv; ++j) {
    int a = j % max_val;
    int b = (j >> bit_width) % max_val;
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

    svec.push_back(sample);
  }

  BinaryAdditionSample batch;
  const int max_in = (bit_width << 1) + 2;
  for (int i = 0; i < max_in; ++i) {
    for (size_t j = 0U; j < svec.size(); ++j) {
      batch.input_tokens.push_back(svec[j].input_tokens[i]);
    }
  }
  const int max_out = bit_width + 1;
  for (int i = 0; i < max_out; ++i) {
    for (size_t j = 0U; j < svec.size(); ++j) {
      batch.target_bits.push_back(svec[j].target_bits[i]);
    }
  }

  return batch;
}

int main() {
  const tcapint bit_width = 2;
  const tcapint seq_len = (bit_width << 1U) + 2U; // a + b =
  const tcapint target_len = bit_width + 1U;      // sum bits
  const tcapint vocab_size = 5;

  const tcapint d_model = 8;
  const tcapint d_ff = 16;
  const tcapint num_heads = 1;

  const int epochs = 100;

  // ---- Model ----
  Sequential model(
      {std::make_shared<Embedding>(vocab_size, d_model),
       std::make_shared<LearnedPositionalEncoding>(seq_len, d_model),
       std::make_shared<TransformerEncoderLayer>(d_model, num_heads, d_ff),
       std::make_shared<Linear>(d_model, 1)});

  std::vector<ParameterPtr> params = model.parameters();

  Adam optimizer(R(0.01));
  optimizer.register_parameters(params);

  auto sample = generate_samples(bit_width);
  const tcapint batch_size = sample.target_bits.size() / target_len;

  auto input = std::make_shared<SymbolTensor>(
      sample.input_tokens, std::vector<tcapint>{batch_size, seq_len});

  auto target = std::make_shared<Tensor>(
      sample.target_bits, std::vector<tcapint>{batch_size, target_len});

  // ---- Training ----
  size_t epoch = 1;
  real1 loss_r = ONE_R1;
  while ((epoch <= epochs) && (loss_r > 0.01)) {
    auto logits = model.forward(input);
    logits->squeeze(2);

    // We take only last (target_len) positions
    auto predicted = Tensor::slice(logits, 1, seq_len - target_len, target_len);

    auto loss = bci_with_logits_loss(predicted, target);

    Tensor::backward(loss);
    adam_step(optimizer, params);

    loss_r = GET_REAL(loss);
    if ((epoch % 10) == 0U) {
      std::cout << "Epoch " << epoch << ", Loss: " << loss_r << std::endl;
    }

    zero_grad(params);
    ++epoch;
  }

  return 0;
}
