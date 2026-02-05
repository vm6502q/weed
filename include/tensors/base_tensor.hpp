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
 * Non-mathematical tensor, solely for indexing (by integer enumeration)
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
};
} // namespace Weed
