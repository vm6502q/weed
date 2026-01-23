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

struct Tensor : public std::enable_shared_from_this<Tensor> {
  StoragePtr storage;

  std::vector<vecCapIntGpu> shape;
  std::vector<vecCapIntGpu> stride;
  vecCapIntGpu offset;

  NodePtr grad_node;
  TensorPtr grad;

  Tensor()
      : storage(nullptr), shape(), stride(), offset(0U), grad_node(nullptr),
        grad(nullptr) {}
  Tensor(std::vector<vecCapIntGpu> shp, std::vector<vecCapIntGpu> strd,
         bool rg = false, DType dtype = DType::REAL,
         DeviceTag dtag = DeviceTag::CPU, int64_t did = -1);

  bool requires_grad() { return !!grad; }

  TensorPtr get_ptr() { return shared_from_this(); }

  virtual vecCapIntGpu get_size() const {
    if (shape.empty()) {
      return 0U;
    }
    vecCapIntGpu max_index = offset;
    for (size_t i = 0; i < shape.size(); ++i) {
      max_index += (shape[i] - 1U) * stride[i];
    }

    return max_index + 1U;
  }

  // Shallow copy:
  Tensor copy() {
    Tensor cp;
    // A tensor is a view on storage:
    cp.storage = storage;
    cp.shape = shape;
    cp.stride = stride;
    cp.offset = offset;
    cp.grad_node = grad_node;
    cp.grad = grad;

    return cp;
  }

  void upcast(DType dt) { storage = storage->Upcast(dt); }

  Tensor operator[](size_t idx);

  static DType get_dtype_by_presidence(const Tensor &left,
                                       const Tensor &right) {
    if (right.storage->dtype == DType::COMPLEX) {
      return DType::COMPLEX;
    }
    return left.storage->dtype;
  }

  static Tensor allocate_like(const Tensor &orig, const DType &dt,
                              const bool &rg);
  static Tensor allocate_like(const std::vector<vecCapIntGpu> &shape,
                              const std::vector<vecCapIntGpu> &stride,
                              const Tensor &orig, const DType &dt,
                              const bool &rg);

  static void backward(Tensor &loss);

  static Tensor transpose(Tensor &a);

  static Tensor abs(Tensor &a);
  static void make_abs_node(Tensor &a, Tensor &out);

  static Tensor relu(Tensor &a);
  static void make_relu_node(Tensor &a, Tensor &out);

  static Tensor add(Tensor &a, Tensor &b);
  static void make_add_node(Tensor &a, Tensor &b, Tensor &out);

  static Tensor mul(Tensor &a, Tensor &b);
  static void make_mul_node(Tensor &a, Tensor &b, Tensor &out);

  static Tensor matmul(Tensor &a, Tensor &b);
  static void make_matmul_node(Tensor &a, Tensor &b, Tensor &out);
};

inline Tensor operator+(Tensor &left, Tensor &right) {
  return Tensor::add(left, right);
}
inline Tensor operator*(Tensor &left, Tensor &right) {
  return Tensor::mul(left, right);
}
} // namespace Weed
