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

#include "common/weed_types.hpp"
#include "device_tag.hpp"
#include "dtype.hpp"
#include "storage.hpp"

#include <vector>

namespace Weed {
struct Tensor;

typedef std::shared_ptr<Tensor> TensorPtr;

struct Tensor {
  StoragePtr storage;

  std::vector<vecCapInt> shape;
  std::vector<vecCapInt> stride;
  vecCapInt offset;

  NodePtr grad_node;
  TensorPtr grad;

  Tensor()
      : storage(nullptr), shape(), stride(), offset(ZERO_VCI),
        grad_node(nullptr), grad(nullptr) {}
  Tensor(std::vector<vecCapInt> shp, std::vector<vecCapInt> strd,
         bool rg = false, DType dtype = DType::REAL,
         DeviceTag dtag = DeviceTag::CPU, int64_t did = -1);
  Tensor(std::vector<real1> val, std::vector<vecCapInt> shp,
         std::vector<vecCapInt> strd, bool rg = false,
         DeviceTag dtag = DeviceTag::CPU, int64_t did = -1);
  Tensor(std::vector<complex> val, std::vector<vecCapInt> shp,
         std::vector<vecCapInt> strd, bool rg = false,
         DeviceTag dtag = DeviceTag::CPU, int64_t did = -1);

  bool requires_grad() { return !!grad; }

  virtual vecCapInt get_size() const {
    if (shape.empty()) {
      return ZERO_VCI;
    }
    vecCapInt max_index = offset;
    for (size_t i = 0U; i < shape.size(); ++i) {
      max_index += (shape[i] - ONE_VCI) * stride[i];
    }

    return max_index + ONE_VCI;
  }

  // Shallow copy:
  TensorPtr copy() {
    TensorPtr cp = std::make_shared<Tensor>();
    // A tensor is a view on storage:
    cp->storage = storage;
    cp->shape = shape;
    cp->stride = stride;
    cp->offset = offset;
    cp->grad_node = grad_node;
    cp->grad = grad;

    return cp;
  }

  void upcast(DType dt) { storage = storage->Upcast(dt); }

  TensorPtr operator[](vecCapInt idx);

  static DType get_dtype_by_presidence(const TensorPtr left,
                                       const TensorPtr right) {
    if (right->storage->dtype == DType::COMPLEX) {
      return DType::COMPLEX;
    }
    return left->storage->dtype;
  }

  static TensorPtr allocate_like(const TensorPtr orig, const DType &dt,
                                 const bool &rg);
  static TensorPtr allocate_like(const std::vector<vecCapInt> &shape,
                                 const std::vector<vecCapInt> &stride,
                                 const TensorPtr orig, const DType &dt,
                                 const bool &rg);

  static void backward(TensorPtr loss);

  static TensorPtr transpose(TensorPtr a);

  static TensorPtr abs(TensorPtr a);
  static void make_abs_node(TensorPtr a, TensorPtr out);

  static TensorPtr relu(TensorPtr a);
  static void make_relu_node(TensorPtr a, TensorPtr out);

  static TensorPtr add(TensorPtr a, TensorPtr b);
  static void make_add_node(TensorPtr a, TensorPtr b, TensorPtr out);

  static TensorPtr mul(TensorPtr a, TensorPtr b);
  static void make_mul_node(TensorPtr a, TensorPtr b, TensorPtr out);

  static TensorPtr matmul(TensorPtr a, TensorPtr b);
  static void make_matmul_node(TensorPtr a, TensorPtr b, TensorPtr out);
};

inline TensorPtr operator+(TensorPtr left, TensorPtr right) {
  return Tensor::add(left, right);
}
inline TensorPtr operator*(TensorPtr left, TensorPtr right) {
  return Tensor::mul(left, right);
}
inline TensorPtr operator>>(TensorPtr left, TensorPtr right) {
  return Tensor::matmul(left, right);
}
inline TensorPtr operator<<(TensorPtr right, TensorPtr left) {
  return Tensor::matmul(left, right);
}
} // namespace Weed
