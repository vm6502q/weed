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

#include "enums/device_tag.hpp"
#include "enums/dtype.hpp"
#include "tensors/base_tensor.hpp"

#include <vector>

#define SCALAR(v, o)                                                           \
  std::make_shared<Tensor>(v, false, o->storage->device,                       \
                           o->storage->get_device_id())

namespace Weed {
struct Tensor;

typedef std::shared_ptr<Tensor> TensorPtr;

/**
 * Tensor with arbitrary dimensions and autograd
 */
struct Tensor : public BaseTensor {
  NodePtr grad_node;
  TensorPtr grad;
  bool requires_grad;

  std::vector<bool> freeze;

  Tensor() {}
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
  Tensor(const Tensor &orig) { copy(orig); }

  void validate_dtype(const DType &dtype) {
    if (dtype == DType::INT) {
      throw std::invalid_argument("Tensor cannot have DType::INT! (INT is only "
                                  "for SymbolTensor, not arithmetic Tensor.)");
    }
  }

  void freeze_init_broadcast() {
    if (stride.size() == 1U) {
      // Never freeze a single (broadcast) index
      // (or else the index isn't broadcast anyway)
      return;
    }

    for (size_t i = 0U; i < stride.size(); ++i) {
      // Freeze all initial broadcast indices
      freeze[i] = !stride[i];
    }
  }

  /**
   * Make this tensor a shallow copy of another
   */
  void copy(const Tensor &cp) {
    // A tensor is a view on storage:
    BaseTensor::copy(cp);
    freeze = cp.freeze;
    grad_node = cp.grad_node;
    grad = cp.grad;
    requires_grad = cp.requires_grad;
  }

  void make_gradient(const bool &force_sparse = false);

  /**
   * For broadcast, make this scalar match the shape of a target Tensor
   */
  bool match_shape(const TensorPtr a);

  /**
   * Materialize all broadcast indices
   */
  void materialize_broadcast();

  /**
   * For internal use, sum the gradient over all broadcast indices
   */
  void reduce_grad_broadcast();

  /**
   * Select a sub-tensor from the position in the outermost tensor index
   */
  TensorPtr operator[](const tcapint &idx) const;

  /**
   * Internally cast this real-value tensor to a complex-value tensor (if
   * necessary)
   */
  void upcast(const DType &dt) { storage = storage->Upcast(dt); }

  /**
   * Cast this CPU-based tensor to a GPU-based one tensor or vice-versa (if
   * necessary)
   */
  TensorPtr cast(const DeviceTag &dt) const {
    TensorPtr cp = std::make_shared<Tensor>(*this);
    if (dt == DeviceTag::CPU) {
      cp->storage = cp->storage->cpu();
    } else if (dt == DeviceTag::GPU) {
      cp->storage = cp->storage->gpu();
    }

    return cp;
  }

  /**
   * Split tensor into equally-sized chunks along axis
   */
  std::vector<TensorPtr> chunk(const size_t &chunks, const int64_t &axis = -1);

  /**
   * A view into a contiguous sub-range of a Tensor along one axis
   */
  TensorPtr slice(const int64_t &axis, const tcapint &start,
                  const tcapint &length);

  /**
   * Tensor initialized with 0
   */
  static TensorPtr zeros(const std::vector<tcapint> &shape,
                         const bool &rg = false,
                         const DType &dtype = DType::REAL,
                         const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
                         const int64_t &did = -1, const bool &s = true) {
    TensorPtr z = std::make_shared<Tensor>(shape, full_contiguous_stride(shape),
                                           rg, dtype, dtag, did, s);
    z->storage->FillZeros();

    return z;
  }

  /**
   * Tensor initialized with 1
   */
  static TensorPtr ones_like(const std::vector<tcapint> &shape,
                             const bool &rg = false,
                             const DType &dtype = DType::REAL,
                             const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
                             const int64_t &did = -1, const bool &s = true) {
    TensorPtr z = std::make_shared<Tensor>(shape, full_contiguous_stride(shape),
                                           rg, dtype, dtag, did, s);
    z->storage->FillOnes();

    return z;
  }

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
   * Compare the device of two tensors and return the higher-performance one
   */
  static DeviceTag get_dtag_by_presidence(const std::vector<TensorPtr> &v);

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

  /**
   * Make a gradient tensor (static)
   */
  static TensorPtr make_gradient(const std::vector<tcapint> &shp,
                                 const DType &dtype, const DeviceTag &dtag,
                                 const int64_t did, const bool &s) {
    // This must be reduced along broadcast dimensions
    // during the backward() step.
    TensorPtr g = std::make_shared<Tensor>(shp, full_contiguous_stride(shp),
                                           false, dtype, dtag, did, s);
    g->storage->FillZeros();

    return g;
  }

  /**
   * Create a new Tensor like the original, but a Scalar, and without Storage
   * value initialization
   */
  static TensorPtr allocate_scalar_like(const Tensor &orig, const bool &rg);

  /**
   * Create a new Tensor like the original, without Storage value initialization
   */
  static TensorPtr allocate_like(const Tensor &orig, const DType &dt,
                                 const bool &rg, const bool &s);

  /**
   * Create a new Tensor like the original, without Storage value initialization
   */
  static TensorPtr allocate_like(const std::vector<tcapint> &shape,
                                 const Tensor &orig, const DType &dt,
                                 const bool &rg, const bool &s);

  /**
   * Create a new Tensor like the original, without Storage value initialization
   */
  static TensorPtr allocate_like(const std::vector<tcapint> &shape,
                                 const std::vector<tcapint> &stride,
                                 const Tensor &orig, const DType &dt,
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
   * Average of all elements by axis (with autograd)
   */
  static TensorPtr mean(TensorPtr a, const tcapint &axis) {
    return div(sum(a, axis), SCALAR((real1)a->shape[axis], a));
  }

  /**
   * Sum of all elements by axis (with autograd)
   */
  static TensorPtr sum(TensorPtr a, const tcapint &axis);
  static void make_sum_node(TensorPtr a, TensorPtr out, const tcapint &axis);

  /**
   * Maximum of all elements by axis (with autograd)
   */
  static TensorPtr max(TensorPtr a, const tcapint &axis);
  /**
   * Minimum of all elements by axis (with autograd)
   */
  static TensorPtr min(TensorPtr a, const tcapint &axis);
  static void make_match_node(TensorPtr a, TensorPtr out, const tcapint &axis);

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
   * Hyberbolic tangent activation function (with autograd)
   */
  static TensorPtr tanh(TensorPtr a);
  static void make_tanh_node(TensorPtr a, TensorPtr out);

  /**
   * Maximum (real) extremum
   */
  static TensorPtr max(TensorPtr a);
  static void make_max_node(TensorPtr a, TensorPtr out);

  /**
   * Minimum (real) extremum
   */
  static TensorPtr min(TensorPtr a);
  static void make_min_node(TensorPtr a, TensorPtr out);

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
  return Tensor::add(SCALAR(left, right), right);
}
inline TensorPtr operator+(TensorPtr left, real1 right) { return right + left; }
inline TensorPtr operator+(complex left, TensorPtr right) {
  return Tensor::add(SCALAR(left, right), right);
}
inline TensorPtr operator+(TensorPtr left, complex right) {
  return right + left;
}
inline TensorPtr operator-(TensorPtr left, TensorPtr right) {
  return Tensor::sub(left, right);
}
inline TensorPtr operator-(real1 left, TensorPtr right) {
  return Tensor::sub(SCALAR(left, right), right);
}
inline TensorPtr operator-(TensorPtr left, real1 right) {
  return Tensor::sub(left, SCALAR(right, left));
}
inline TensorPtr operator-(complex left, TensorPtr right) {
  return Tensor::sub(SCALAR(left, right), right);
}
inline TensorPtr operator-(TensorPtr left, complex right) {
  return Tensor::sub(left, SCALAR(right, left));
}
inline TensorPtr operator*(TensorPtr left, TensorPtr right) {
  return Tensor::mul(left, right);
}
inline TensorPtr operator*(real1 left, TensorPtr right) {
  return Tensor::mul(SCALAR(left, right), right);
}
inline TensorPtr operator*(TensorPtr left, real1 right) { return right * left; }
inline TensorPtr operator*(complex left, TensorPtr right) {
  return Tensor::mul(SCALAR(left, right), right);
}
inline TensorPtr operator*(TensorPtr left, complex right) {
  return right * left;
}
inline TensorPtr operator/(TensorPtr left, TensorPtr right) {
  return Tensor::div(left, right);
}
inline TensorPtr operator/(real1 left, TensorPtr right) {
  return Tensor::div(SCALAR(left, right), right);
}
inline TensorPtr operator/(TensorPtr left, real1 right) {
  return Tensor::div(left, SCALAR(right, left));
}
inline TensorPtr operator/(complex left, TensorPtr right) {
  return Tensor::div(SCALAR(left, right), right);
}
inline TensorPtr operator/(TensorPtr left, complex right) {
  return Tensor::div(left, SCALAR(right, left));
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
