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

  std::vector<tcapint> shape;
  std::vector<tcapint> stride;
  tcapint offset;

  NodePtr grad_node;
  TensorPtr grad;

  Tensor()
      : storage(nullptr), shape(), stride(), offset(ZERO_VCI),
        grad_node(nullptr), grad(nullptr) {}
  Tensor(std::vector<tcapint> shp, std::vector<tcapint> strd, bool rg = false,
         DType dtype = DType::REAL, DeviceTag dtag = DeviceTag::DEFAULT_DEVICE,
         int64_t did = -1, bool s = true, bool gs = true);
  Tensor(const std::vector<real1> &val, std::vector<tcapint> shp,
         std::vector<tcapint> strd, bool rg = false,
         DeviceTag dtag = DeviceTag::DEFAULT_DEVICE, int64_t did = -1);
  Tensor(const std::vector<complex> &val, std::vector<tcapint> shp,
         std::vector<tcapint> strd, bool rg = false,
         DeviceTag dtag = DeviceTag::DEFAULT_DEVICE, int64_t did = -1);
  Tensor(real1 val, bool rg = false, DeviceTag dtag = DeviceTag::DEFAULT_DEVICE,
         int64_t did = -1)
      : Tensor(std::vector<real1>{val}, std::vector<tcapint>{1},
               std::vector<tcapint>{0}, rg, dtag, did) {}
  Tensor(complex val, bool rg = false,
         DeviceTag dtag = DeviceTag::DEFAULT_DEVICE, int64_t did = -1)
      : Tensor(std::vector<complex>{val}, std::vector<tcapint>{1},
               std::vector<tcapint>{0}, rg, dtag, did) {}
  Tensor(const RealSparseVector &val, std::vector<tcapint> shp,
         std::vector<tcapint> strd, bool rg = false);
  Tensor(const ComplexSparseVector &val, std::vector<tcapint> shp,
         std::vector<tcapint> strd, bool rg = false);

  bool validate_shape(const std::vector<tcapint> &shp,
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
   * Will we calculate gradients on back-propagation?
   */
  bool requires_grad() { return !!grad; }

  /**
   * How many elements are in this tensor?
   */
  virtual tcapint get_size() const {
    if (shape.empty()) {
      return ZERO_VCI;
    }
    tcapint max_index = offset;
    for (size_t i = 0U; i < shape.size(); ++i) {
      max_index += (shape[i] - ONE_VCI) * stride[i];
    }

    return max_index + 1U;
  }

  /**
   * How many elements are broadcast in this tensor?
   */
  virtual tcapint get_broadcast_size() const {
    if (shape.empty()) {
      return ZERO_VCI;
    }
    tcapint max_index = 0U;
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
  void match_shape(const TensorPtr a);

  void reduce_grad_broadcast();

  /**
   * Select a sub-tensor from the position in the outermost tensor index
   */
  TensorPtr operator[](tcapint idx);

  /**
   * Compare the data type of two tensors and return the more-encompassing one
   */
  static DType get_dtype_by_presidence(const std::vector<TensorPtr> v) {
    for (const TensorPtr &p : v) {
      if (p->storage->dtype == DType::COMPLEX) {
        return DType::COMPLEX;
      }
    }
    return DType::REAL;
  }

  /**
   * Ensure that all tensors in a list are on the same device
   */
  static bool all_same_device(const std::vector<TensorPtr> &);

  /**
   * Create a new Tensor like the original, without Storage value initialization
   */
  static TensorPtr allocate_like(const TensorPtr orig, const DType &dt,
                                 const bool &rg, const bool &s,
                                 const bool &gs = false);
  /**
   * Create a new Tensor like the original, without Storage value initialization
   */
  static TensorPtr allocate_like(const std::vector<tcapint> &shape,
                                 const std::vector<tcapint> &stride,
                                 const TensorPtr orig, const DType &dt,
                                 const bool &rg, const bool &s,
                                 const bool &gs = false);

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
   * Sum of all elements (with autograd)
   */
  static TensorPtr sum(TensorPtr a);
  static void make_sum_node(TensorPtr a, TensorPtr out);

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
   * Element-wise clamp
   */
  static TensorPtr clamp(TensorPtr a, real1 lo, real1 hi);
  static void make_clamp_node(TensorPtr a, real1 lo, real1 hi, TensorPtr out);

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
  static TensorPtr exp(TensorPtr a, real1 b = E_R1);
  static void make_exp_node(TensorPtr a, TensorPtr log_b, TensorPtr out);

  /**
   * Element-wise logarithm
   */
  static TensorPtr log(TensorPtr a, real1 b = E_R1);
  static void make_log_node(TensorPtr a, TensorPtr inv_log_b, TensorPtr out);
};

inline TensorPtr operator+(TensorPtr left, TensorPtr right) {
  return Tensor::add(left, right);
}
inline TensorPtr operator+(real1 left, TensorPtr right) {
  TensorPtr l = std::make_shared<Tensor>(left, right->requires_grad(),
                                         right->storage->device,
                                         right->storage->get_device_id());
  return Tensor::add(l, right);
}
inline TensorPtr operator+(TensorPtr left, real1 right) { return right + left; }
inline TensorPtr operator-(TensorPtr left, TensorPtr right) {
  return Tensor::sub(left, right);
}
inline TensorPtr operator-(real1 left, TensorPtr right) {
  TensorPtr l = std::make_shared<Tensor>(left, right->requires_grad(),
                                         right->storage->device,
                                         right->storage->get_device_id());
  return Tensor::sub(l, right);
}
inline TensorPtr operator-(TensorPtr left, real1 right) {
  TensorPtr r = std::make_shared<Tensor>(right, left->requires_grad(),
                                         left->storage->device,
                                         left->storage->get_device_id());
  return Tensor::sub(left, r);
}
inline TensorPtr operator*(TensorPtr left, TensorPtr right) {
  return Tensor::mul(left, right);
}
inline TensorPtr operator*(real1 left, TensorPtr right) {
  TensorPtr l = std::make_shared<Tensor>(left, right->requires_grad(),
                                         right->storage->device,
                                         right->storage->get_device_id());
  return Tensor::mul(l, right);
}
inline TensorPtr operator*(TensorPtr left, real1 right) { return right * left; }
inline TensorPtr operator/(TensorPtr left, TensorPtr right) {
  return Tensor::div(left, right);
}
inline TensorPtr operator/(real1 left, TensorPtr right) {
  TensorPtr l = std::make_shared<Tensor>(left, right->requires_grad(),
                                         right->storage->device,
                                         right->storage->get_device_id());
  return Tensor::div(l, right);
}
inline TensorPtr operator/(TensorPtr left, real1 right) {
  TensorPtr r = std::make_shared<Tensor>(right, left->requires_grad(),
                                         left->storage->device,
                                         left->storage->get_device_id());
  return Tensor::div(left, r);
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
inline TensorPtr operator^(real1 base, TensorPtr power) {
  return Tensor::exp(power, base);
}
} // namespace Weed
