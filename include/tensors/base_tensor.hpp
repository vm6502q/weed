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

#include "storage/storage.hpp"

#include <vector>

namespace Weed {
struct BaseTensor;
typedef std::shared_ptr<BaseTensor> BaseTensorPtr;
/**
 * Base tensor class for both symbolic (SymbolTensor) and mathematical (Tensor)
 * objects
 */
struct BaseTensor {
  StoragePtr storage;
  tcapint offset;
  std::vector<tcapint> shape;
  std::vector<tcapint> stride;

  BaseTensor() : storage(nullptr), offset(0U), shape(), stride() {}
  BaseTensor(const std::vector<tcapint> &shp, const std::vector<tcapint> &strd)
      : storage(nullptr), offset(0U), shape(shp), stride(strd) {
    validate_constructor();
  }

  virtual ~BaseTensor() {}

  /**
   * Make this (base) tensor a shallow copy of another
   */
  void copy(const BaseTensor &cp) {
    // A tensor is a view on storage:
    storage = cp.storage;
    offset = cp.offset;
    shape = cp.shape;
    stride = cp.stride;
  }

  /**
   * Validate the constructor parameters
   */
  virtual void validate_constructor() {
    if (shape.size() != stride.size()) {
      throw std::invalid_argument(
          "Tensor shape vector must have same length as stride vector!");
    }

    if ((shape.size() == 1U) && (shape[0U] == 1U)) {
      stride[0U] = 0U;
    } else if (!is_contiguous()) {
      throw std::invalid_argument(
          "Initial tensor shape and stride must be contiguous!");
    }
  }

  /**
   * How many elements are in this tensor?
   */
  tcapint get_size() const {
    if (shape.empty()) {
      return 0U;
    }

    tcapint max_index = 0U;
    for (size_t i = 0U; i < shape.size(); ++i) {
      max_index += (shape[i] - 1U) * stride[i];
    }

    return max_index + 1U;
  }

  /**
   * How many elements are broadcast in this tensor?
   */
  tcapint get_broadcast_size() const {
    if (shape.empty()) {
      return 0U;
    }

    tcapint max_index = 1U;
    for (size_t i = 0U; i < shape.size(); ++i) {
      max_index *= shape[i];
    }

    return max_index;
  }

  /**
   * Is the Tensor Storage contiguous (i.e., densely packed in a traversable
   * order)?
   */
  bool is_contiguous() const { return !offset && is_contiguous(shape, stride); }

  /**
   * Is this Tensor a Scalar (i.e., has only a single storage element that's
   * broadcast)?
   */
  bool is_scalar() const {
    if (shape.empty()) {
      return false;
    }

    for (size_t i = 0U; i < shape.size(); ++i) {
      if (((shape[i] - 1U) * stride[i]) != 0U) {
        return false;
      }
    }

    return true;
  }

  /**
   *  Look up an index in Storage based on shape and stride
   */
  tcapint get_storage_index(const tcapint &idx) const {
    if (is_scalar()) {
      return offset;
    }

    tcapint curr = idx;
    tcapint stor = offset;

    for (size_t i = 0U; (i < shape.size()) && curr; ++i) {
      const tcapint &l = shape[i];
      stor += (curr % l) * stride[i];
      curr /= l;
    }

    if (curr) {
      throw std::invalid_argument("Tensor index out-of-range!");
    }

    return stor;
  }

  /**
   * Reshape the tensor
   */
  void reshape(const std::vector<symint> &s) {
    if (!is_contiguous(shape, stride)) {
      throw std::domain_error(
          "Can't reshape BaseTensor that isn't contiguous!");
    }

    const tcapint total = get_size();

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

    shape = resolved;
    stride = full_contiguous_stride(resolved);
  }

  /**
   * If the tensor has exactly two indices, transpose them
   */
  void transpose() {
    if (shape.size() > 2U) {
      throw std::invalid_argument(
          "Tensor::transpose is only for 2D tensors (and "
          "vectors and covectors)!");
    }

    if (shape.size() == 1U) {
      // Treat input as column vector, and transpose to row vector
      shape = {1U, shape[0U]};
      stride = {0U, stride[0U]};
    } else {
      std::swap(shape[0U], shape[1U]);
      std::swap(stride[0U], stride[1U]);
    }
  }

  /**
   * Transpose the two tensor indices
   */
  void transpose(symint i, symint j) {
    while (i < 0) {
      i += shape.size();
    }
    while (j < 0) {
      j += shape.size();
    }

    if (i != j) {
      std::swap(shape[i], shape[j]);
      std::swap(stride[i], stride[j]);
    }
  }

  /**
   * Validate the Tensor shape, for constructors
   */
  static bool is_contiguous(const std::vector<tcapint> &shp,
                            const std::vector<tcapint> &s) {
    tcapint st = 1U;
    for (size_t i = 0U; i < s.size(); ++i) {
      if (!s[i]) {
        continue;
      }
      if (s[i] != st) {
        return false;
      }
      st *= shp[i];
    }

    return true;
  }

  /**
   * Find the gradient stride (before reduction), for constructors
   */
  static std::vector<tcapint>
  full_contiguous_stride(const std::vector<tcapint> &shp) {
    if ((shp.size() == 1U) && (shp[0U] == 1U)) {
      return std::vector<tcapint>{0U};
    }

    std::vector<tcapint> g_stride(shp.size());
    tcapint max_index = 1U;
    for (size_t i = 0U; i < shp.size(); ++i) {
      g_stride[i] = max_index;
      max_index *= shp[i];
    }

    return g_stride;
  }
};
} // namespace Weed
