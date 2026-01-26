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
#include "ops/clamp.hpp"
#include "ops/commuting.hpp"
#include "ops/div.hpp"
#include "ops/in_place.hpp"
#include "ops/matmul.hpp"
#include "ops/pow.hpp"
#include "ops/real_unary.hpp"
#include "ops/reduce.hpp"
#include "ops/sub.hpp"
#include "ops/sum.hpp"
#include "tensors/real_scalar.hpp"

#include "storage/all_storage.hpp"

#include <unordered_set>

#define GET_REAL(ptr) static_cast<RealScalar *>((ptr).get())->get_item()

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
  int64_t did = storage_ptr->get_device_id();

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
  if (!validate_shape(shape, stride)) {
    throw std::invalid_argument(
        "Initial tensor shape and stride must be contiguous!");
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
  if (!validate_shape(shape, stride)) {
    throw std::invalid_argument(
        "Initial tensor shape and stride must be contiguous!");
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
    return std::make_shared<Scalar>(v);
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

void Tensor::match_shape(const TensorPtr a) {
  shape = a->shape;
  stride.resize(shape.size());

  if (requires_grad()) {
    // This must be reduced along broadcast dimensions
    // uring the backward() step.
    TensorPtr g = allocate_like(a, storage->dtype, false);
    g->storage->FillZeros();
    grad->match_shape(g);
    Weed::add_in_place(*(g.get()), *(grad.get()));
    grad = g;
  }
}

void Tensor::reduce_grad_broadcast() {
  if (!requires_grad()) {
    return;
  }

  for (int64_t i = stride.size() - 1U; i >= 0; --i) {
    if (stride[i]) {
      continue;
    }

    std::vector<vecCapIntGpu> sh = grad->shape;
    std::vector<vecCapIntGpu> st = grad->stride;

    if (sh.size() == 1U) {
      sh[0U] = 1U;
      st[0U] = 0U;
    } else {
      sh.erase(sh.begin() + i);
      st.erase(st.begin() + i);
    }

    TensorPtr tmp = allocate_like(sh, st, grad, grad->storage->dtype, false);
    Weed::reduce(i, *(grad.get()), *(tmp.get()));
    grad = tmp;
  }
}

void Tensor::backward(TensorPtr loss) {
  if (!loss || !loss->requires_grad()) {
    return;
  }

  // Seed gradient
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
        dfs(p->grad_node);
      }
    }
    topo.push_back(n);
  };

  dfs(loss->grad_node);

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

TensorPtr Tensor::sum(TensorPtr a) {
  const bool rg = a->requires_grad();
  TensorPtr out =
      allocate_like(std::vector<vecCapInt>{1U}, std::vector<vecCapInt>{0U}, a,
                    a->storage->dtype, rg);

  Weed::sum(*(a.get()), *(out.get()));

  if (rg) {
    make_sum_node(a, out);
  }

  return out;
}

void Tensor::make_sum_node(TensorPtr a, TensorPtr out) {
  out->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a}, [a, out]() {
        TensorPtr a_grad = a->grad;
        TensorPtr out_grad = out->grad;
        TensorPtr scale =
            std::make_shared<RealScalar>(ONE_R1, false, out->storage->device);
        // da += dout  (broadcast)
        const DType &dt = get_dtype_by_presidence(a_grad, out_grad);
        a_grad->upcast(dt);
        TensorPtr tmp = Tensor::allocate_like(out_grad, dt, false);
        scale->match_shape(out_grad);
        Weed::mul(*(out_grad.get()), *(scale.get()), *(tmp.get()));
        Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
        a->reduce_grad_broadcast();
      });
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
        const DType &dt = get_dtype_by_presidence(a_grad, out_grad);
        a_grad->upcast(dt);
        scale->match_shape(out_grad);
        TensorPtr tmp = Tensor::allocate_like(out_grad, dt, false);
        Weed::mul(*(out_grad.get()), *(scale.get()), *(tmp.get()));
        Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
        a->reduce_grad_broadcast();
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
        Tensor &a_grad = *(a->grad.get());
        a_grad.upcast(out_grad.storage->dtype);
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
        Tensor &a_grad = *(a->grad.get());
        a_grad.upcast(out_grad.storage->dtype);
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
        Tensor &a_grad = *(a->grad.get());
        a_grad.upcast(out_grad.storage->dtype);
        Weed::relu_grad(a_grad, *(a.get()), out_grad);
      });
}

TensorPtr Tensor::clamp(TensorPtr a, real1 lo, real1 hi) {
  const bool rg = a->requires_grad();
  TensorPtr out = allocate_like(a, a->storage->dtype, rg);

  Weed::clamp(*(a.get()), lo, hi, *(out.get()));

  if (rg) {
    make_clamp_node(a, lo, hi, out);
  }

  return out;
}

void Tensor::make_clamp_node(TensorPtr a, real1 lo, real1 hi, TensorPtr out) {
  out->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a}, [a, out, lo, hi]() {
        TensorPtr dx = a->grad;
        TensorPtr dy = out->grad;
        dx->upcast(dy->storage->dtype);
        dy->upcast(dx->storage->dtype);
        Weed::clamp_grad(*(dy.get()), *(a.get()), lo, hi, *(dx.get()));
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
    if (a->requires_grad()) {
      a->match_shape(b);
      TensorPtr a_grad = a->grad;
      a_grad->upcast(out_grad->storage->dtype);
      Weed::add_in_place(*(a_grad.get()), *(out_grad.get()));
      a->reduce_grad_broadcast();
    }
    if (b->requires_grad()) {
      b->match_shape(a);
      TensorPtr b_grad = b->grad;
      b_grad->upcast(out_grad->storage->dtype);
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
        if (a->requires_grad()) {
          a->match_shape(b);
          TensorPtr a_grad = a->grad;
          const DType &dt = get_dtype_by_presidence(a_grad, out_grad);
          a_grad->upcast(dt);
          TensorPtr tmp = Tensor::allocate_like(b, dt, false);
          Weed::mul(*(out_grad.get()), *(b.get()), *(tmp.get()));
          Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
          a->reduce_grad_broadcast();
        }
        if (b->requires_grad()) {
          b->match_shape(a);
          TensorPtr b_grad = b->grad;
          const DType &dt = get_dtype_by_presidence(b_grad, out_grad);
          b_grad->upcast(dt);
          TensorPtr tmp = Tensor::allocate_like(a, dt, false);
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
        if (a->requires_grad()) {
          TensorPtr a_grad = a->grad;
          const DType &dt = get_dtype_by_presidence(a_grad, out_grad);
          a_grad->upcast(dt);
          TensorPtr bt = transpose(b);
          TensorPtr tmp = Tensor::allocate_like(a, dt, false);
          Weed::matmul(*(out->grad.get()), *(bt.get()), *(tmp.get()));
          Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
        }
        if (b->requires_grad()) {
          TensorPtr b_grad = b->grad;
          const DType &dt = get_dtype_by_presidence(b_grad, out_grad);
          b_grad->upcast(dt);
          TensorPtr at = transpose(a);
          TensorPtr tmp = Tensor::allocate_like(b, dt, false);
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
    if (a->requires_grad()) {
      a->match_shape(b);
      TensorPtr a_grad = a->grad;
      a_grad->upcast(out_grad->storage->dtype);
      Weed::add_in_place(*(a_grad.get()), *(out_grad.get()));
      a->reduce_grad_broadcast();
    }
    if (b->requires_grad()) {
      b->match_shape(a);
      TensorPtr b_grad = b->grad;
      b_grad->upcast(out_grad->storage->dtype);
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
        if (a->requires_grad()) {
          a->match_shape(b);
          TensorPtr a_grad = a->grad;
          const DType &dt = get_dtype_by_presidence(a_grad, out_grad);
          a_grad->upcast(dt);
          TensorPtr tmp = Tensor::allocate_like(b, dt, false);
          Weed::div(*(out_grad.get()), *(b.get()), *(tmp.get()));
          Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
          a->reduce_grad_broadcast();
        }
        if (b->requires_grad()) {
          b->match_shape(a);
          TensorPtr b_grad = b->grad;
          const DType &dt = get_dtype_by_presidence(b_grad, out_grad);
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

TensorPtr Tensor::pow(TensorPtr a, real1 p) {
  const bool rg = a->requires_grad();
  TensorPtr out = allocate_like(a, a->storage->dtype, rg);

  Weed::pow(*(a.get()), p, *(out.get()));

  if (rg) {
    RealScalarPtr y = std::make_shared<RealScalar>(p, false, a->storage->device,
                                                   a->storage->get_device_id());
    make_pow_node(a, y, out);
  }

  return out;
}

void Tensor::make_pow_node(TensorPtr x, TensorPtr p, TensorPtr y) {
  y->grad_node = std::make_shared<Node>(std::vector<TensorPtr>{x}, [x, p, y]() {
    TensorPtr dx = x->grad;
    TensorPtr dy = y->grad;

    TensorPtr dy_y = Tensor::allocate_like(dy, dy->storage->dtype, false);
    Weed::mul(*(dy.get()), *(y.get()), *(dy_y.get()));

    TensorPtr dy_y_p = Tensor::allocate_like(dy_y, dy_y->storage->dtype, false);
    p->match_shape(dy_y);
    Weed::mul(*(dy_y.get()), *(p.get()), *(dy_y_p.get()));

    TensorPtr r = Tensor::allocate_like(dy_y_p, dy_y_p->storage->dtype, false);
    Weed::div(*(dy_y_p.get()), *(x.get()), *(r.get()));

    dx->upcast(r->storage->dtype);
    Weed::add_in_place(*(dx.get()), *(r.get()));
  });
}

TensorPtr Tensor::exp(TensorPtr a, real1 b) {
  const bool rg = a->requires_grad();
  TensorPtr out = allocate_like(a, a->storage->dtype, rg);

  Weed::exp(*(a.get()), b, *(out.get()));

  if (rg) {
    RealScalarPtr y = std::make_shared<RealScalar>((real1)std::log(b), false,
                                                   a->storage->device,
                                                   a->storage->get_device_id());
    make_exp_node(a, y, out);
  }

  return out;
}

void Tensor::make_exp_node(TensorPtr x, TensorPtr log_b, TensorPtr y) {
  y->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{x}, [x, log_b, y]() {
        TensorPtr dx = x->grad;
        TensorPtr dy = y->grad;

        TensorPtr dy_v = Tensor::allocate_like(dy, dy->storage->dtype, false);
        log_b->match_shape(dy);
        Weed::mul(*(dy.get()), *(log_b.get()), *(dy_v.get()));

        TensorPtr r = Tensor::allocate_like(dy_v, dy_v->storage->dtype, false);
        Weed::mul(*(dy_v.get()), *(y.get()), *(r.get()));

        dx->upcast(r->storage->dtype);
        Weed::add_in_place(*(dx.get()), *(r.get()));
      });
}

TensorPtr Tensor::log(TensorPtr a, real1 b) {
  const bool rg = a->requires_grad();
  TensorPtr out = allocate_like(a, a->storage->dtype, rg);

  Weed::log(*(a.get()), b, *(out.get()));

  if (rg) {
    RealScalarPtr y = std::make_shared<RealScalar>((real1)(1.0 / std::log(b)),
                                                   false, a->storage->device,
                                                   a->storage->get_device_id());
    make_log_node(a, y, out);
  }

  return out;
}

void Tensor::make_log_node(TensorPtr x, TensorPtr inv_log_b, TensorPtr y) {
  y->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{x}, [x, inv_log_b, y]() {
        TensorPtr dx = x->grad;
        TensorPtr dy = y->grad;

        TensorPtr dy_v = Tensor::allocate_like(dy, dy->storage->dtype, false);
        inv_log_b->match_shape(dy);
        Weed::mul(*(dy.get()), *(inv_log_b.get()), *(dy_v.get()));

        TensorPtr r = Tensor::allocate_like(dy_v, dy_v->storage->dtype, false);
        Weed::div(*(dy_v.get()), *(y.get()), *(r.get()));

        dx->upcast(r->storage->dtype);
        Weed::add_in_place(*(dx.get()), *(r.get()));
      });
}
} // namespace Weed
