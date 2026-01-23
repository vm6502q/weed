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

#include "tensor.hpp"

namespace Weed {
struct Scalar;

typedef std::shared_ptr<Scalar> ScalarPtr;

struct Scalar : public Tensor {
  Scalar(real1 v, bool rg, DeviceTag dtag, int64_t did = -1)
      : Tensor(std::vector<real1>{v}, std::vector<vecCapInt>{ONE_VCI},
               std::vector<vecCapInt>{ZERO_VCI}, rg, dtag, did) {}
  Scalar(complex v, bool rg, DeviceTag dtag, int64_t did = -1)
      : Tensor(std::vector<complex>{v}, std::vector<vecCapInt>{ONE_VCI},
               std::vector<vecCapInt>{ZERO_VCI}, rg, dtag, did) {}
  Scalar(TensorPtr orig) {
    if (orig->get_size() != ONE_VCI) {
      throw std::invalid_argument(
          "Cannot construct scalar from Tensor with get_size() != 1!");
    }
    shape = std::vector<vecCapInt>{ONE_VCI};
    stride = std::vector<vecCapInt>{ZERO_VCI};
    offset = orig->offset;
    storage = orig->storage;
    grad_node = orig->grad_node;
    grad = orig->grad;
  }

  void set_index_count(size_t ic) {
    shape.resize(ic);
    stride.resize(ic);
  }
  void reset_indices() {
    set_index_count(1U);
    shape[0U] = ONE_VCI;
  }
  void match_shape(const TensorPtr a) {
    shape = a->shape;
    const size_t sz = shape.size();
    stride.resize(sz);
    if (requires_grad()) {
      grad->shape = a->shape;
      grad->stride.resize(sz);
    }
  }

  static ScalarPtr allocate_like(const ScalarPtr orig, const DType &dt,
                                 const bool &rg) {
    return std::make_shared<Scalar>(Tensor::allocate_like(orig, dt, rg));
  }

  static ScalarPtr abs(ScalarPtr a);
  static ScalarPtr relu(ScalarPtr a);
  static ScalarPtr add(ScalarPtr a, ScalarPtr b);
  static ScalarPtr mul(ScalarPtr a, ScalarPtr b);

  static TensorPtr add(ScalarPtr a, TensorPtr b);
  static TensorPtr mul(ScalarPtr a, TensorPtr b);
};

inline ScalarPtr operator+(ScalarPtr left, ScalarPtr right) {
  return Scalar::add(left, right);
}
inline TensorPtr operator+(ScalarPtr left, TensorPtr right) {
  return Scalar::add(left, right);
}
inline TensorPtr operator+(TensorPtr &left, ScalarPtr right) {
  return Scalar::add(right, left);
}
inline ScalarPtr operator*(ScalarPtr left, ScalarPtr right) {
  return Scalar::mul(left, right);
}
inline TensorPtr operator*(ScalarPtr left, TensorPtr right) {
  return Scalar::mul(left, right);
}
inline TensorPtr operator*(TensorPtr left, ScalarPtr right) {
  return Scalar::mul(right, left);
}
} // namespace Weed
