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

#include "tensors/base_tensor.hpp"

namespace Weed {
struct SymbolTensor;
typedef std::shared_ptr<SymbolTensor> SymbolTensorPtr;

/**
 * Non-mathematical tensor, solely for indexing (by integer enumeration)
 */
struct SymbolTensor : BaseTensor {
  SymbolTensor(const std::vector<tcapint> &shp,
               const std::vector<tcapint> &strd, const bool &rg = false,
               const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
               const int64_t &did = -1, const bool &s = true);
  SymbolTensor(const std::vector<symint> &val, const std::vector<tcapint> &shp,
               const std::vector<tcapint> &strd, const bool &rg = false,
               const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
               const int64_t &did = -1);

  /**
   * Split tensor into equally-sized chunks along axis
   */
  static std::vector<SymbolTensorPtr>
  chunk(SymbolTensorPtr a, const size_t &chunks, int64_t axis = -1) {
    // Contributed by Elara (the OpenAI custom GPT)
    if (chunks == 0) {
      throw std::invalid_argument("Tensor::chunk: chunks must be > 0");
    }

    if (axis < 0) {
      axis += a->shape.size();
    }
    if (axis < 0 || axis >= (int64_t)a->shape.size()) {
      throw std::invalid_argument("Tensor::chunk: axis out of range");
    }

    const tcapint dim = a->shape[axis];
    if (dim % chunks != 0) {
      throw std::invalid_argument(
          "Tensor::chunk: dimension not divisible by chunks");
    }

    const tcapint chunk_dim = dim / chunks;

    std::vector<SymbolTensorPtr> out;
    out.reserve(chunks);

    for (size_t i = 0; i < chunks; ++i) {
      SymbolTensorPtr t = std::make_shared<SymbolTensor>(
          *(a.get())); // shallow copy (shared storage)

      t->shape[axis] = chunk_dim;
      t->offset += i * chunk_dim * a->stride[axis];

      out.push_back(t);
    }

    return out;
  }

  /**
   * Reshape the tensor
   */
  static SymbolTensorPtr reshape(const SymbolTensorPtr a,
                                 const std::vector<symint> &s) {
    const tcapint total = a->get_size();

    // Resolve -1
    std::vector<tcapint> resolved;
    resolved.reserve(s.size());
    for (size_t i = 0U; i < s.size(); ++i) {
      resolved.push_back((tcapint)s[i]);
    }

    symint infer_index = -1;
    tcapint known_product = 1U;

    for (size_t i = 0U; i < s.size(); ++i) {
      if (s[i] < 0) {
        if (infer_index != -1) {
          throw std::invalid_argument(
              "Tensor::reshape(): only one -1 dimension allowed");
        }
        infer_index = (symint)i;
      } else {
        known_product *= s[i];
      }
    }

    if (infer_index >= 0) {
      if ((!known_product) || (total % known_product)) {
        throw std::invalid_argument(
            "Tensor::reshape(): cannot infer dimension size");
      }
      resolved[infer_index] = total / known_product;
    }

    // Final size check
    tcapint new_size = 1;
    for (tcapint d : resolved) {
      new_size *= d;
    }

    if (new_size != total) {
      throw std::invalid_argument("Tensor::reshape(): sizes do not match");
    }

    SymbolTensorPtr out = std::make_shared<SymbolTensor>(*(a.get()));
    out->shape = resolved;
    out->stride = full_contiguous_stride(resolved);

    return out;
  }

  /**
   * If the tensor has exactly two indices, transpose them
   */
  static SymbolTensorPtr transpose(const SymbolTensorPtr a) {
    if (a->shape.size() > 2U) {
      throw std::invalid_argument(
          "Tensor::transpose is only for 2D tensors (and "
          "vectors and covectors)!");
    }

    SymbolTensorPtr out = std::make_shared<SymbolTensor>(*(a.get()));

    if (out->shape.size() == 1U) {
      // Treat input as column vector, and transpose to row vector
      out->shape = {1U, out->shape[0U]};
      out->stride = {0U, out->stride[0U]};

      return out;
    }

    std::swap(out->shape[0U], out->shape[1U]);
    std::swap(out->stride[0U], out->stride[1U]);

    return out;
  }

  /**
   * Transpose the two tensor indices
   */
  static SymbolTensorPtr transpose(const SymbolTensorPtr a, symint i,
                                   symint j) {
    while (i < 0) {
      i += a->shape.size();
    }
    while (j < 0) {
      j += a->shape.size();
    }

    SymbolTensorPtr out = std::make_shared<SymbolTensor>(*(a.get()));

    if (i == j) {
      return out;
    }

    std::swap(out->shape[i], out->shape[j]);
    std::swap(out->stride[i], out->stride[j]);

    return out;
  }
};
} // namespace Weed
