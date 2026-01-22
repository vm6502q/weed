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
      : Tensor(std::vector<vecCapIntGpu>{1U}, std::vector<vecCapIntGpu>{0U}, rg,
               DType::REAL, dtag, did) {}
  Scalar(complex v, bool rg, DeviceTag dtag, int64_t did = -1)
      : Tensor(std::vector<vecCapIntGpu>{1U}, std::vector<vecCapIntGpu>{0U}, rg,
               DType::COMPLEX, dtag, did) {}

  void set_index_count(size_t ic) {
    shape.resize(ic);
    stride.resize(ic);
  }
  void reset_indices() {
    set_index_count(1U);
    shape[0U] = 1U;
  }
  void match_shape(const Tensor &a) {
    shape = a.shape;
    const size_t sz = shape.size();
    stride.resize(sz);
    if (requires_grad()) {
      grad->shape = a.shape;
      grad->stride.resize(sz);
    }
  }

  static Scalar allocate_like(const Scalar &orig, const DType &dt,
                              const bool &rg);

  static Scalar abs(Scalar &a);
  static Scalar relu(Scalar &a);
  static Scalar add(Scalar &a, Scalar &b);
  static Scalar mul(Scalar &a, Scalar &b);

  static Tensor add(Scalar &a, Tensor &b);
  static Tensor mul(Scalar &a, Tensor &b);
};

inline Scalar operator+(Scalar &left, Scalar &right) {
  return Scalar::add(left, right);
}
inline Tensor operator+(Scalar &left, Tensor &right) {
  return Scalar::add(left, right);
}
inline Tensor operator+(Tensor &left, Scalar &right) {
  return Scalar::add(right, left);
}
inline Scalar operator*(Scalar &left, Scalar &right) {
  return Scalar::mul(left, right);
}
inline Tensor operator*(Scalar &left, Tensor &right) {
  return Scalar::mul(left, right);
}
inline Tensor operator*(Tensor &left, Scalar &right) {
  return Scalar::mul(right, left);
}
} // namespace Weed
