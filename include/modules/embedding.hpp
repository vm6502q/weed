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
#include "tensors/symbol_tensor.hpp"

namespace Weed {
/**
 * Embedding module to gather symbols over weights
 */
struct Embedding : public Module {
  tcapint num_embeddings;
  tcapint embedding_dim;
  ParameterPtr weight;

  Embedding() : Module(EMBEDDING_T) {}
  Embedding(const tcapint &vocab, const tcapint &dim,
            const DType &dtype = DType::REAL,
            const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE, int64_t did = -1)
      : Module(EMBEDDING_T), num_embeddings(vocab), embedding_dim(dim),
        weight(std::make_shared<Parameter>(std::vector<tcapint>{vocab, dim},
                                           std::vector<tcapint>{1, vocab}, true,
                                           dtype, dtag, did)) {}
  using Module::forward;
  TensorPtr forward(const TensorPtr) override {
    throw std::domain_error(
        "Embedding::forward(x) takes a SymbolTensor, not a Tensor!");
  }
  TensorPtr forward(const BaseTensorPtr t) override {
    return forward(std::dynamic_pointer_cast<SymbolTensor>(t));
  }
  TensorPtr forward(const SymbolTensorPtr t);
  std::vector<ParameterPtr> parameters() override { return {weight}; }

  void save(std::ostream &) const override;
};
typedef std::shared_ptr<Embedding> EmbeddingPtr;
} // namespace Weed
