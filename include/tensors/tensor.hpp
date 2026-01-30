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
  bool requires_grad;

  Tensor(const std::vector<tcapint> &shp, const std::vector<tcapint> &strd,
         const bool &rg = false, const DType &dtype = DType::REAL,
         const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
         const int64_t &did = -1, const bool &s = true);
  Tensor(const std::vector<real1> &val, const std::vector<tcapint> &shp,
         const std::vector<tcapint> &strd, const bool &rg = false,
         const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
         const int64_t &did = -1);
  Tensor(const std::vector<complex> &val, const std::vector<tcapint> &shp,
         const std::vector<tcapint> &strd, const bool &rg = false,
         const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
         const int64_t &did = -1);
  Tensor(const real1 &val, const bool &rg = false,
         const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
         const int64_t &did = -1)
      : Tensor(std::vector<real1>{val}, std::vector<tcapint>{1U},
               std::vector<tcapint>{0U}, rg, dtag, did) {}
  Tensor(const complex &val, const bool &rg = false,
         const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
         const int64_t &did = -1)
      : Tensor(std::vector<complex>{val}, std::vector<tcapint>{1U},
               std::vector<tcapint>{0U}, rg, dtag, did) {}
  Tensor(const RealSparseVector &val, const std::vector<tcapint> &shp,
         const std::vector<tcapint> &strd, const bool &rg = false);
  Tensor(const ComplexSparseVector &val, const std::vector<tcapint> &shp,
         const std::vector<tcapint> &strd, const bool &rg = false);

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
    tcapint max_index = 1U;
    for (size_t i = 0U; i < shape.size(); ++i) {
      max_index *= shape[i];
    }

    return max_index;
  }

  /**
   * Make a shallow copy of this tensor
   */
  TensorPtr copy() const {
    TensorPtr cp = std::make_shared<Tensor>(
        shape, stride, requires_grad, storage->dtype, storage->device,
        storage->get_device_id(), storage->is_sparse());
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
  void copy(const TensorPtr cp) {
    // A tensor is a view on storage:
    storage = cp->storage;
    shape = cp->shape;
    stride = cp->stride;
    offset = cp->offset;
    grad_node = cp->grad_node;
    grad = cp->grad;
  }

  void make_gradient() {
    if (!requires_grad) {
      throw std::domain_error("Called Tensor::make_gradient() on a node "
                              "instance that does not require autograd!");
    }

    grad =
        Tensor::make_gradient(shape, storage->dtype, storage->device,
                              storage->get_device_id(), storage->is_sparse());
  }

  /**
   * Internally cast this real-value tensor to a complex-value tensor (if
   * necessary)
   */
  void upcast(const DType &dt) { storage = storage->Upcast(dt); }

  /**
   * Make a gradient tensor (static)
   */
  static TensorPtr make_gradient(const std::vector<tcapint> &shp,
                                 const DType &dtype, const DeviceTag &dtag,
                                 const int64_t did, const bool &s) {
    // This must be reduced along broadcast dimensions
    // during the backward() step.
    TensorPtr g = std::make_shared<Tensor>(shp, gradient_stride(shp), false,
                                           dtype, dtag, did, s);
    g->storage->FillZeros();

    return g;
  }

  /**
   * For broadcast, make this scalar match the shape of a target Tensor
   */
  void match_shape(const TensorPtr a);

  void reduce_grad_broadcast();

  /**
   * Select a sub-tensor from the position in the outermost tensor index
   */
  TensorPtr operator[](const tcapint &idx) const;

  /**
   * Compare the data type of two tensors and return the more-encompassing one
   */
  static DType get_dtype_by_presidence(const std::vector<TensorPtr> &v) {
    for (const TensorPtr &p : v) {
      if (p->storage->dtype == DType::COMPLEX) {
        return DType::COMPLEX;
      }
    }
    return DType::REAL;
  }

  /**
   * Validate the Tensor shape, for constructors
   */
  static bool validate_shape(const std::vector<tcapint> &shp,
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
  static std::vector<tcapint> gradient_stride(const std::vector<tcapint> &shp) {
    std::vector<tcapint> g_stride(shp.size());
    tcapint max_index = 1U;
    for (size_t i = 0U; i < shp.size(); ++i) {
      g_stride[i] = max_index;
      max_index *= shp[i];
    }

    return g_stride;
  }
  /**
   * Ensure that all tensors in a list are on the same device
   */
  static bool all_same_device(const std::vector<TensorPtr> &);

  /**
   * Create a new Tensor like the original, but a Scalar, and without Storage
   * value initialization
   */
  static TensorPtr allocate_scalar_like(const TensorPtr orig, const bool &rg);

  /**
   * Create a new Tensor like the original, without Storage value initialization
   */
  static TensorPtr allocate_like(const TensorPtr orig, const DType &dt,
                                 const bool &rg, const bool &s);
  /**
   * Create a new Tensor like the original, without Storage value initialization
   */
  static TensorPtr allocate_like(const std::vector<tcapint> &shape,
                                 const std::vector<tcapint> &stride,
                                 const TensorPtr orig, const DType &dt,
                                 const bool &rg, const bool &s);

  /**
   * Use autograd to calculate gradients that are in the same graph as this
   * Tensor
   */
  static void backward(const TensorPtr loss);

  /**
   * If the tensor has exactly two indices, transpose them
   */
  static TensorPtr transpose(const TensorPtr a);

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
  static void make_pow_node(TensorPtr a, real1 p, TensorPtr out);

  /**
   * Element-wise logarithm
   */
  static TensorPtr exp(TensorPtr a, real1 b = E_R1);
  static void make_exp_node(TensorPtr a, real1 log_b, TensorPtr out);

  /**
   * Element-wise logarithm
   */
  static TensorPtr log(TensorPtr a, real1 b = E_R1);
  static void make_log_node(TensorPtr a, real1 inv_log_b, TensorPtr out);
};

inline TensorPtr operator+(TensorPtr left, TensorPtr right) {
  return Tensor::add(left, right);
}
inline TensorPtr operator+(real1 left, TensorPtr right) {
  TensorPtr l = std::make_shared<Tensor>(left, false, right->storage->device,
                                         right->storage->get_device_id());
  return Tensor::add(l, right);
}
inline TensorPtr operator+(TensorPtr left, real1 right) { return right + left; }
inline TensorPtr operator-(TensorPtr left, TensorPtr right) {
  return Tensor::sub(left, right);
}
inline TensorPtr operator-(real1 left, TensorPtr right) {
  TensorPtr l = std::make_shared<Tensor>(left, false, right->storage->device,
                                         right->storage->get_device_id());
  return Tensor::sub(l, right);
}
inline TensorPtr operator-(TensorPtr left, real1 right) {
  TensorPtr r = std::make_shared<Tensor>(right, false, left->storage->device,
                                         left->storage->get_device_id());
  return Tensor::sub(left, r);
}
inline TensorPtr operator*(TensorPtr left, TensorPtr right) {
  return Tensor::mul(left, right);
}
inline TensorPtr operator*(real1 left, TensorPtr right) {
  TensorPtr l = std::make_shared<Tensor>(left, false, right->storage->device,
                                         right->storage->get_device_id());
  return Tensor::mul(l, right);
}
inline TensorPtr operator*(TensorPtr left, real1 right) { return right * left; }
inline TensorPtr operator/(TensorPtr left, TensorPtr right) {
  return Tensor::div(left, right);
}
inline TensorPtr operator/(real1 left, TensorPtr right) {
  TensorPtr l = std::make_shared<Tensor>(left, false, right->storage->device,
                                         right->storage->get_device_id());
  return Tensor::div(l, right);
}
inline TensorPtr operator/(TensorPtr left, real1 right) {
  TensorPtr r = std::make_shared<Tensor>(right, false, left->storage->device,
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
