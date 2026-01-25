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
#include "enums/device_tag.hpp"
#include "enums/dtype.hpp"
#include "storage/storage.hpp"

#include <vector>

namespace Weed {
struct Tensor;

typedef std::shared_ptr<Tensor> TensorPtr;

/**
 * Tensor with arbitrary dimensions and autograd
 */
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
         DeviceTag dtag = DeviceTag::DEFAULT_DEVICE, int64_t did = -1);
  Tensor(std::vector<real1> val, std::vector<vecCapInt> shp,
         std::vector<vecCapInt> strd, bool rg = false,
         DeviceTag dtag = DeviceTag::DEFAULT_DEVICE, int64_t did = -1);
  Tensor(std::vector<complex> val, std::vector<vecCapInt> shp,
         std::vector<vecCapInt> strd, bool rg = false,
         DeviceTag dtag = DeviceTag::DEFAULT_DEVICE, int64_t did = -1);

  /**
   * Will we calculate gradients on back-propagation?
   */
  bool requires_grad() { return !!grad; }

  /**
   * How many elements are in this tensor?
   */
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

  /**
   * How many elements are broadcast in this tensor?
   */
  virtual vecCapInt get_broadcast_size() const {
    if (shape.empty()) {
      return ZERO_VCI;
    }
    vecCapInt max_index = 0U;
    for (size_t i = 0U; i < shape.size(); ++i) {
      max_index *= shape[i];
    }

    return max_index;
  }

  /**
   * Make a shallow copy of this tensor
   */
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

  /**
   * Make this tensor a shallow copy of another
   */
  void copy(TensorPtr cp) {
    // A tensor is a view on storage:
    storage = cp->storage;
    shape = cp->shape;
    stride = cp->stride;
    offset = cp->offset;
    grad_node = cp->grad_node;
    grad = cp->grad;
  }

  /**
   * Internally cast this real-value tensor to a complex-value tensor (if
   * necessary)
   */
  void upcast(DType dt) { storage = storage->Upcast(dt); }

  /**
   * For broadcast, make this scalar match the shape of a target Tensor
   */
  void match_shape(const TensorPtr a) {
    shape = a->shape;
    stride.resize(shape.size());

    if (requires_grad()) {
      // This must be reduced along broadcast dimensions
      // uring the backward() step.
      grad = allocate_like(a, storage->dtype, false);
      grad->storage->FillZeros();
    }
  }

  void reduce_grad_broadcast();

  /**
   * Select a sub-tensor from the position in the outermost tensor index
   */
  TensorPtr operator[](vecCapInt idx);

  /**
   * Compare the data type of two tensors and return the more-encompassing one
   */
  static DType get_dtype_by_presidence(const TensorPtr left,
                                       const TensorPtr right) {
    if (right->storage->dtype == DType::COMPLEX) {
      return DType::COMPLEX;
    }
    return left->storage->dtype;
  }

  /**
   * Ensure that all tensors in a list are on the same device
   */
  static bool all_same_device(const std::vector<TensorPtr> &);

  /**
   * Create a new Tensor like the original, without Storage value initialization
   */
  static TensorPtr allocate_like(const TensorPtr orig, const DType &dt,
                                 const bool &rg);
  /**
   * Create a new Tensor like the original, without Storage value initialization
   */
  static TensorPtr allocate_like(const std::vector<vecCapInt> &shape,
                                 const std::vector<vecCapInt> &stride,
                                 const TensorPtr orig, const DType &dt,
                                 const bool &rg);

  /**
   * Use autograd to calculate gradients that are in the same graph as this
   * Tensor
   */
  static void backward(TensorPtr loss);

  /**
   * If the tensor has exactly two indices, transpose them
   */
  static TensorPtr transpose(TensorPtr a);

  /**
   * Average of all elements (with autograd)
   */
  static TensorPtr mean(TensorPtr a);
  static void make_mean_node(TensorPtr a, TensorPtr out);

  /**
   * Absolute value (with autograd)
   */
  static TensorPtr abs(TensorPtr a);
  static void make_abs_node(TensorPtr a, TensorPtr out);

  /**
   * Sigmoid activation function (with autograd)
   */
  static TensorPtr sigmoid(TensorPtr a);
  static void make_sigmoid_node(TensorPtr a, TensorPtr out);

  /**
   * Rectified linear activation function (with autograd)
   */
  static TensorPtr relu(TensorPtr a);
  static void make_relu_node(TensorPtr a, TensorPtr out);

  /**
   * Element-wise addition (with autograd)
   */
  static TensorPtr add(TensorPtr a, TensorPtr b);
  static void make_add_node(TensorPtr a, TensorPtr b, TensorPtr out);

  /**
   * Element-wise multiplication (with autograd)
   */
  static TensorPtr mul(TensorPtr a, TensorPtr b);
  static void make_mul_node(TensorPtr a, TensorPtr b, TensorPtr out);

  /**
   * Matrix multiplication (with autograd)
   */
  static TensorPtr matmul(TensorPtr a, TensorPtr b);
  static void make_matmul_node(TensorPtr a, TensorPtr b, TensorPtr out);

  /**
   * Element-wise subtraction (with autograd)
   */
  static TensorPtr sub(TensorPtr a, TensorPtr b);
  static void make_sub_node(TensorPtr a, TensorPtr b, TensorPtr out);

  /**
   * Element-wise division (with autograd)
   */
  static TensorPtr div(TensorPtr a, TensorPtr b);
  static void make_div_node(TensorPtr a, TensorPtr b, TensorPtr out);

  /**
   * Element-wise power
   */
  static TensorPtr pow(TensorPtr a, real1 p);
  static void make_pow_node(TensorPtr a, TensorPtr p, TensorPtr out);

  /**
   * Element-wise logarithm
   */
  static TensorPtr log(TensorPtr a, real1 b = E_R1);
  static void make_log_node(TensorPtr a, TensorPtr inv_log_b, TensorPtr out);
};

inline TensorPtr operator+(TensorPtr left, TensorPtr right) {
  return Tensor::add(left, right);
}
inline TensorPtr operator-(TensorPtr left, TensorPtr right) {
  return Tensor::sub(left, right);
}
inline TensorPtr operator*(TensorPtr left, TensorPtr right) {
  return Tensor::mul(left, right);
}
inline TensorPtr operator/(TensorPtr left, TensorPtr right) {
  return Tensor::div(left, right);
}
inline TensorPtr operator>>(TensorPtr left, TensorPtr right) {
  return Tensor::matmul(left, right);
}
inline TensorPtr operator<<(TensorPtr right, TensorPtr left) {
  return Tensor::matmul(left, right);
}
inline TensorPtr operator^(TensorPtr base, real1 power) {
  return Tensor::pow(base, power);
}
} // namespace Weed
