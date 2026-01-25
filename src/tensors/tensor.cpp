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

#include "autograd/node.hpp"
#include "ops/abs.hpp"
#include "ops/commuting.hpp"
#include "ops/div.hpp"
#include "ops/in_place.hpp"
#include "ops/matmul.hpp"
#include "ops/mean.hpp"
#include "ops/real_unary.hpp"
#include "ops/reduce.hpp"
#include "ops/sub.hpp"
#include "tensors/complex_scalar.hpp"
#include "tensors/real_scalar.hpp"

#include "storage/all_storage.hpp"

#include <unordered_set>

#include <iostream>

#define INIT_DEVICE_STORAGE(val, GpuType, CpuType)                             \
  switch (dtag) {                                                              \
  case DeviceTag::GPU:                                                         \
    storage = std::make_shared<GpuType>(val, did);                             \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    storage = std::make_shared<CpuType>(val);                                  \
  }

namespace Weed {
bool Tensor::all_same_device(const std::vector<TensorPtr> &t) {
  if (t.empty()) {
    return true;
  }

  const DeviceTag d = t[0U]->storage->device;
  for (size_t i = 1U; i < t.size(); ++i) {
    if (d != t[i]->storage->device) {
      return false;
    }
  }

  return true;
}

TensorPtr Tensor::allocate_like(const TensorPtr orig, const DType &dt,
                                const bool &rg) {
  const StoragePtr storage_ptr = orig->storage;
  const DeviceTag dtag = storage_ptr->device;
  const int64_t did = storage_ptr->get_device_id();

  return std::make_shared<Tensor>(orig->shape, orig->stride, rg, dt, dtag, did);
}

TensorPtr Tensor::allocate_like(const std::vector<vecCapInt> &shape,
                                const std::vector<vecCapInt> &stride,
                                const TensorPtr orig, const DType &dt,
                                const bool &rg) {
  const StoragePtr storage_ptr = orig->storage;
  const DeviceTag dtag = storage_ptr->device;
  int64_t did = -1;
#if ENABLE_GPU
  if (dtag == DeviceTag::GPU) {
    switch (dt) {
    case DType::COMPLEX:
      did = static_cast<GpuComplexStorage *>(storage_ptr.get())->gpu->deviceID;
      break;
    case DType::REAL:
    default:
      did = static_cast<GpuRealStorage *>(storage_ptr.get())->gpu->deviceID;
    }
  }
#endif
  return std::make_shared<Tensor>(shape, stride, rg, dt, dtag, did);
}

Tensor::Tensor(std::vector<vecCapInt> shp, std::vector<vecCapInt> strd, bool rg,
               DType dtype, DeviceTag dtag, int64_t did)
    : shape(shp), stride(strd), offset(ZERO_VCI), grad_node(nullptr),
      grad(rg ? std::make_shared<Tensor>(shp, strd, false, dtype, dtag, did)
              : nullptr) {
  if (shape.size() != stride.size()) {
    throw std::invalid_argument(
        "Tensor shape vector must have same length as stride vector!");
  }

  const vecCapInt size = get_size();

  switch (dtype) {
  case DType::COMPLEX:
#if ENABLE_GPU
    INIT_DEVICE_STORAGE(size, GpuComplexStorage, CpuComplexStorage);
#else
    storage = std::make_shared<CpuComplexStorage>(size);
#endif
    break;
  case DType::REAL:
  default:
#if ENABLE_GPU
    INIT_DEVICE_STORAGE(size, GpuRealStorage, CpuRealStorage);
#else
    storage = std::make_shared<CpuRealStorage>(size);
#endif
  }

  if (rg) {
    grad->storage->FillZeros();
  }
}

Tensor::Tensor(std::vector<real1> val, std::vector<vecCapInt> shp,
               std::vector<vecCapInt> strd, bool rg, DeviceTag dtag,
               int64_t did)
    : shape(shp), stride(strd), offset(ZERO_VCI), grad_node(nullptr),
      grad(rg ? std::make_shared<Tensor>(shp, strd, false, DType::REAL, dtag,
                                         did)
              : nullptr) {
  if (shape.size() != stride.size()) {
    throw std::invalid_argument(
        "Tensor shape vector must have same length as stride vector!");
  }

  const vecCapInt size = get_size();

  if (size != val.size()) {
    throw std::invalid_argument("Tensor value initializer vector must have "
                                "same size as implied by shape and stride!");
  }

#if ENABLE_GPU
  INIT_DEVICE_STORAGE(val, GpuRealStorage, CpuRealStorage);
#else
  storage = std::make_shared<CpuRealStorage>(val);
#endif

  if (rg) {
    grad->storage->FillZeros();
  }
}
Tensor::Tensor(std::vector<complex> val, std::vector<vecCapInt> shp,
               std::vector<vecCapInt> strd, bool rg, DeviceTag dtag,
               int64_t did)
    : shape(shp), stride(strd), offset(ZERO_VCI), grad_node(nullptr),
      grad(rg ? std::make_shared<Tensor>(shp, strd, false, DType::COMPLEX, dtag,
                                         did)
              : nullptr) {
  if (shape.size() != stride.size()) {
    throw std::invalid_argument(
        "Tensor shape vector must have same length as stride vector!");
  }

  const vecCapInt size = get_size();

  if (size != val.size()) {
    throw std::invalid_argument("Tensor value initializer vector must have "
                                "same size as implied by shape and stride!");
  }

#if ENABLE_GPU
  INIT_DEVICE_STORAGE(val, GpuComplexStorage, CpuComplexStorage);
#else
  storage = std::make_shared<CpuComplexStorage>(val);
#endif

  if (rg) {
    grad->storage->FillZeros();
  }
}

TensorPtr Tensor::operator[](vecCapInt idx) {
  if (idx > shape.back()) {
    throw std::invalid_argument("Tensor index out-of-range!");
  }

  TensorPtr v = copy();
  v->offset += idx * stride.back();
  v->shape.pop_back();
  v->stride.pop_back();

  if (v->grad) {
    v->grad = (*(v->grad))[idx];
  }

  if (v->shape.empty()) {
    switch (v->storage->dtype) {
    case DType::COMPLEX:
      return std::make_shared<ComplexScalar>(v);
    case DType::REAL:
      return std::make_shared<RealScalar>(v);
    default:
      return std::make_shared<Scalar>(v);
    }
  }

  return v;
}

std::vector<TensorPtr> filterParents(std::vector<TensorPtr> parents) {
  std::vector<TensorPtr> filtered;
  for (TensorPtr p : parents) {
    if (p->requires_grad()) {
      filtered.push_back(p);
    }
  }

  return filtered;
}

void Tensor::reduce_grad_broadcast() {
  if ((stride.size() < 2) || !requires_grad()) {
    return;
  }

  for (int64_t i = stride.size() - 1U; i >= 0; --i) {
    if (stride[i]) {
      continue;
    }

    std::vector<vecCapIntGpu> sh = grad->shape;
    std::vector<vecCapIntGpu> st = grad->stride;
    sh.erase(sh.begin() + i);
    st.erase(st.begin() + i);
    TensorPtr tmp = allocate_like(sh, st, grad, grad->storage->dtype, false);
    Weed::reduce(i, *(grad.get()), *(tmp.get()));
    grad = tmp;
  }
}

void Tensor::backward(TensorPtr loss) {
  if (!loss->requires_grad()) {
    return;
  }

  // Seed gradient of loss (scalar assumed)
  // loss.grad already allocated by our invariant
  loss->grad->storage->FillOnes();

  std::vector<NodePtr> topo;
  std::unordered_set<Node *> seen;

  std::function<void(const NodePtr &)> dfs = [&](const NodePtr &n) {
    if (!n || seen.count(n.get())) {
      return;
    }
    seen.insert(n.get());
    for (auto &p : n->parents) {
      if (p && p->grad_node) {
        p->grad_node->backward();
      }
    }
    topo.push_back(n);
  };

  loss->grad_node->backward();

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->backward();
  }
}

TensorPtr Tensor::transpose(TensorPtr a) {
  if (a->shape.size() != 2U) {
    throw std::invalid_argument(
        "Tensor::tranpose is (currently) only for matrices with 2 indices!");
  }

  // Shallow copy (keeps storage and gradient)
  TensorPtr out = a->copy();
  // Change tensor view:
  std::swap(out->shape[0U], out->shape[1U]);
  std::swap(out->stride[0U], out->stride[1U]);

  return out;
}

TensorPtr Tensor::mean(TensorPtr a) {
  const bool rg = a->requires_grad();
  TensorPtr out =
      allocate_like(std::vector<vecCapInt>{1U}, std::vector<vecCapInt>{0U}, a,
                    a->storage->dtype, rg);

  Weed::mean(*(a.get()), *(out.get()));

  if (rg) {
    make_mean_node(a, out);
  }

  return out;
}

void Tensor::make_mean_node(TensorPtr a, TensorPtr out) {
  out->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a}, [a, out]() {
        TensorPtr a_grad = a->grad;
        TensorPtr out_grad = out->grad;
        TensorPtr scale = std::make_shared<RealScalar>(
            ONE_R1 / (real1)a->get_size(), false, out->storage->device);
        // da += dout / N   (broadcast)
        const DType &dt = out_grad->storage->dtype;
        a_grad->upcast(dt);
        TensorPtr tmp = Tensor::allocate_like(out_grad, dt, false);
        Weed::mul(*(out_grad.get()), *(scale.get()), *(tmp.get()));
        Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
      });
}

TensorPtr Tensor::abs(TensorPtr a) {
  const bool rg = a->requires_grad();
  TensorPtr out = allocate_like(a, DType::REAL, rg);

  Weed::abs(*(a.get()), *(out.get()));

  if (rg) {
    make_abs_node(a, out);
  }

  return out;
}

void Tensor::make_abs_node(TensorPtr a, TensorPtr out) {
  out->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a}, [a, out]() {
        Tensor &out_grad = *(out->grad.get());
        const DType &dt = out_grad.storage->dtype;
        Tensor &a_grad = *(a->grad.get());
        a_grad.upcast(dt);
        Weed::abs_grad(a_grad, *(a.get()), out_grad);
      });
}

TensorPtr Tensor::sigmoid(TensorPtr a) {
  const bool rg = a->requires_grad();
  TensorPtr out = allocate_like(a, a->storage->dtype, rg);

  Weed::sigmoid(*(a.get()), *(out.get()));

  if (rg) {
    make_sigmoid_node(a, out);
  }

  return out;
}

void Tensor::make_sigmoid_node(TensorPtr a, TensorPtr out) {
  out->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a}, [a, out]() {
        Tensor &out_grad = *(out->grad.get());
        const DType &dt = out_grad.storage->dtype;
        Tensor &a_grad = *(a->grad.get());
        a_grad.upcast(dt);
        Weed::sigmoid_grad(a_grad, *(a.get()), out_grad);
      });
}

TensorPtr Tensor::relu(TensorPtr a) {
  const bool rg = a->requires_grad();
  TensorPtr out = allocate_like(a, a->storage->dtype, rg);

  Weed::relu(*(a.get()), *(out.get()));

  if (rg) {
    make_relu_node(a, out);
  }

  return out;
}

void Tensor::make_relu_node(TensorPtr a, TensorPtr out) {
  out->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a}, [a, out]() {
        Tensor &out_grad = *(out->grad.get());
        const DType &dt = out_grad.storage->dtype;
        Tensor &a_grad = *(a->grad.get());
        a_grad.upcast(dt);
        Weed::relu_grad(a_grad, *(a.get()), out_grad);
      });
}

TensorPtr Tensor::add(TensorPtr a, TensorPtr b) {
  if (!all_same_device({a, b})) {
    throw std::invalid_argument(
        "Cannot mix Tensor devices in Tensor::add(a, b)!");
  }

  const bool rg = a->requires_grad() || b->requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  TensorPtr out;
  if (a->get_size() == ONE_VCI) {
    a->match_shape(b);
    out = allocate_like(b, dt, rg);
  } else if (b->get_size() == ONE_VCI) {
    b->match_shape(a);
    out = allocate_like(a, dt, rg);
  } else {
    out = allocate_like(a, dt, rg);
  }

  Weed::add(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_add_node(a, b, out);
  }

  return out;
}

void Tensor::make_add_node(TensorPtr a, TensorPtr b, TensorPtr out) {
  out->grad_node = std::make_shared<Node>(filterParents({a, b}), [a, b, out]() {
    TensorPtr out_grad = out->grad;
    const DType &dt = out_grad->storage->dtype;
    if (a->requires_grad()) {
      TensorPtr a_grad = a->grad;
      a_grad->upcast(dt);
      Weed::add_in_place(*(a_grad.get()), *(out_grad.get()));
      a->reduce_grad_broadcast();
    }
    if (b->requires_grad()) {
      TensorPtr b_grad = b->grad;
      b_grad->upcast(dt);
      Weed::add_in_place(*(b_grad.get()), *(out_grad.get()));
      b->reduce_grad_broadcast();
    }
  });
}

TensorPtr Tensor::mul(TensorPtr a, TensorPtr b) {
  if (!all_same_device({a, b})) {
    throw std::invalid_argument(
        "Cannot mix Tensor devices in Tensor::mul(a, b)!");
  }

  const bool rg = a->requires_grad() || b->requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  TensorPtr out;
  if (a->get_size() == ONE_VCI) {
    a->match_shape(b);
    out = allocate_like(b, dt, rg);
  } else if (b->get_size() == ONE_VCI) {
    b->match_shape(a);
    out = allocate_like(a, dt, rg);
  } else {
    out = allocate_like(a, dt, rg);
  }

  Weed::mul(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_mul_node(a, b, out);
  }

  return out;
}

void Tensor::make_mul_node(TensorPtr a, TensorPtr b, TensorPtr out) {
  out->grad_node = std::make_shared<Node>(
      filterParents(std::vector<TensorPtr>{a, b}), [a, b, out]() {
        TensorPtr out_grad = out->grad;
        const DType &dt = out_grad->storage->dtype;
        if (a->requires_grad()) {
          TensorPtr tmp = Tensor::allocate_like(b, dt, false);
          TensorPtr a_grad = a->grad;
          a_grad->upcast(dt);
          Weed::mul(*(out_grad.get()), *(b.get()), *(tmp.get()));
          Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
          a->reduce_grad_broadcast();
        }
        if (b->requires_grad()) {
          TensorPtr tmp = Tensor::allocate_like(a, dt, false);
          TensorPtr b_grad = b->grad;
          b_grad->upcast(dt);
          Weed::mul(*(out_grad.get()), *(a.get()), *(tmp.get()));
          Weed::add_in_place(*(b_grad.get()), *(tmp.get()));
          b->reduce_grad_broadcast();
        }
      });
}

TensorPtr Tensor::matmul(TensorPtr a, TensorPtr b) {
  if (!all_same_device({a, b})) {
    throw std::invalid_argument(
        "Cannot mix Tensor devices in Tensor::matmul(a, b)!");
  }

  if ((a->shape.size() != 2U) || (b->shape.size() != 2U)) {
    throw std::invalid_argument(
        "Tensor::matmul is only for matrices with 2 indices!");
  }

  const std::vector<vecCapInt> shp = {a->shape[0U], b->shape[1U]};
  const std::vector<vecCapInt> str = {ONE_VCI, a->shape[0U]};
  const bool rg = a->requires_grad() || b->requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  TensorPtr out = allocate_like(shp, str, a, dt, rg);

  Weed::matmul(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_matmul_node(a, b, out);
  }

  return out;
}

void Tensor::make_matmul_node(TensorPtr a, TensorPtr b, TensorPtr out) {
  out->grad_node = std::make_shared<Node>(
      filterParents(std::vector<TensorPtr>{a, b}), [a, b, out]() {
        TensorPtr out_grad = out->grad;
        const DType &dt = out_grad->storage->dtype;
        if (a->requires_grad()) {
          TensorPtr bt = transpose(b);
          TensorPtr tmp = Tensor::allocate_like(a, dt, false);
          TensorPtr a_grad = a->grad;
          a_grad->upcast(dt);
          Weed::matmul(*(out->grad.get()), *(bt.get()), *(tmp.get()));
          Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
        }
        if (b->requires_grad()) {
          TensorPtr at = transpose(a);
          TensorPtr tmp = Tensor::allocate_like(b, dt, false);
          TensorPtr b_grad = b->grad;
          b_grad->upcast(dt);
          Weed::matmul(*(at.get()), *(out->grad.get()), *(tmp.get()));
          Weed::add_in_place(*(b_grad.get()), *(tmp.get()));
        }
      });
}

TensorPtr Tensor::sub(TensorPtr a, TensorPtr b) {
  if (!all_same_device({a, b})) {
    throw std::invalid_argument(
        "Cannot mix Tensor devices in Tensor::div(a, b)!");
  }

  const bool rg = a->requires_grad() || b->requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  TensorPtr out;
  if (a->get_size() == ONE_VCI) {
    a->match_shape(b);
    out = allocate_like(b, dt, rg);
  } else if (b->get_size() == ONE_VCI) {
    b->match_shape(a);
    out = allocate_like(a, dt, rg);
  } else {
    out = allocate_like(a, dt, rg);
  }

  Weed::sub(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_sub_node(a, b, out);
  }

  return out;
}

void Tensor::make_sub_node(TensorPtr a, TensorPtr b, TensorPtr out) {
  out->grad_node = std::make_shared<Node>(filterParents({a, b}), [a, b, out]() {
    TensorPtr out_grad = out->grad;
    const DType &dt = out_grad->storage->dtype;
    if (a->requires_grad()) {
      TensorPtr a_grad = a->grad;
      a_grad->upcast(dt);
      Weed::add_in_place(*(a_grad.get()), *(out_grad.get()));
      a->reduce_grad_broadcast();
    }
    if (b->requires_grad()) {
      TensorPtr b_grad = b->grad;
      b_grad->upcast(dt);
      Weed::sub_in_place(*(b_grad.get()), *(out_grad.get()));
      b->reduce_grad_broadcast();
    }
  });
}

TensorPtr Tensor::div(TensorPtr a, TensorPtr b) {
  if (!all_same_device({a, b})) {
    throw std::invalid_argument(
        "Cannot mix Tensor devices in Tensor::div(a, b)!");
  }

  const bool rg = a->requires_grad() || b->requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  TensorPtr out;
  if (a->get_size() == ONE_VCI) {
    a->match_shape(b);
    out = allocate_like(b, dt, rg);
  } else if (b->get_size() == ONE_VCI) {
    b->match_shape(a);
    out = allocate_like(a, dt, rg);
  } else {
    out = allocate_like(a, dt, rg);
  }

  Weed::div(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_div_node(a, b, out);
  }

  return out;
}

void Tensor::make_div_node(TensorPtr a, TensorPtr b, TensorPtr out) {
  out->grad_node = std::make_shared<Node>(
      filterParents(std::vector<TensorPtr>{a, b}), [a, b, out]() {
        TensorPtr out_grad = out->grad;
        const DType &dt = out_grad->storage->dtype;
        if (a->requires_grad()) {
          TensorPtr a_grad = a->grad;
          a_grad->upcast(dt);
          TensorPtr tmp = Tensor::allocate_like(b, dt, false);
          Weed::div(*(out_grad.get()), *(b.get()), *(tmp.get()));
          Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
          a->reduce_grad_broadcast();
        }
        if (b->requires_grad()) {
          TensorPtr b_grad = b->grad;
          b_grad->upcast(dt);
          TensorPtr b_sqr = Tensor::allocate_like(b, b->storage->dtype, false);
          Weed::mul(*(b.get()), *(b.get()), *(b_sqr.get()));
          TensorPtr tmp = Tensor::allocate_like(a, dt, false);
          Weed::div(*(a.get()), *(b_sqr.get()), *(tmp.get()));
          Weed::sub_in_place(*(b_grad.get()), *(tmp.get()));
          b->reduce_grad_broadcast();
        }
      });
}
} // namespace Weed
