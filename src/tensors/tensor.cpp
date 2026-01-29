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
#define IS_SPARSE(a)                                                           \
  (a && a->storage->is_sparse() &&                                             \
   (a->storage->get_sparse_size() <= (a->storage->size >> 1U)))

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
                                const bool &rg, const bool &s, const bool &sg) {
  const StoragePtr storage_ptr = orig->storage;
  const DeviceTag dtag = storage_ptr->device;
  const int64_t did = storage_ptr->get_device_id();

  return std::make_shared<Tensor>(orig->shape, orig->stride, rg, dt, dtag, did,
                                  s, sg);
}

TensorPtr Tensor::allocate_like(const std::vector<tcapint> &shape,
                                const std::vector<tcapint> &stride,
                                const TensorPtr orig, const DType &dt,
                                const bool &rg, const bool &s, const bool &sg) {
  const StoragePtr storage_ptr = orig->storage;
  const DeviceTag dtag = storage_ptr->device;
  int64_t did = storage_ptr->get_device_id();

  return std::make_shared<Tensor>(shape, stride, rg, dt, dtag, did, s, sg);
}

TensorPtr Tensor::allocate_scalar_like(const TensorPtr orig, const bool &rg) {
  StoragePtr st = orig->storage;

  return std::make_shared<Tensor>(
      std::vector<tcapint>{1U}, std::vector<tcapint>{0U}, rg, st->dtype,
      st->device, st->get_device_id(), IS_SPARSE(orig), IS_SPARSE(orig->grad));
}

Tensor::Tensor(std::vector<tcapint> shp, std::vector<tcapint> strd, bool rg,
               DType dtype, DeviceTag dtag, int64_t did, bool s, bool sg)
    : shape(shp), stride(strd), offset(ZERO_VCI), grad_node(nullptr),
      grad(rg ? std::make_shared<Tensor>(shp, strd, false, dtype, dtag, did, sg)
              : nullptr) {
  if (shape.size() != stride.size()) {
    throw std::invalid_argument(
        "Tensor shape vector must have same length as stride vector!");
  }

  const tcapint size = get_size();

  if (s && (dtag == DeviceTag::CPU)) {
    switch (dtype) {
    case DType::COMPLEX:
      storage = std::make_shared<SparseCpuComplexStorage>(size);
      break;
    case DType::REAL:
    default:
      storage = std::make_shared<SparseCpuRealStorage>(size);
    }
  } else {
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
  }

  if (rg) {
    grad->storage->FillZeros();
  }
}

Tensor::Tensor(const std::vector<real1> &val, std::vector<tcapint> shp,
               std::vector<tcapint> strd, bool rg, DeviceTag dtag, int64_t did)
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

  const tcapint size = get_size();

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
Tensor::Tensor(const std::vector<complex> &val, std::vector<tcapint> shp,
               std::vector<tcapint> strd, bool rg, DeviceTag dtag, int64_t did)
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

  const tcapint size = get_size();

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

Tensor::Tensor(const RealSparseVector &val, std::vector<tcapint> shp,
               std::vector<tcapint> strd, bool rg, bool sg)
    : shape(shp), stride(strd), offset(ZERO_VCI), grad_node(nullptr),
      grad(rg ? std::make_shared<Tensor>(shp, strd, false, DType::REAL,
                                         DeviceTag::CPU, -1, sg)
              : nullptr) {
  if (shape.size() != stride.size()) {
    throw std::invalid_argument(
        "Tensor shape vector must have same length as stride vector!");
  }
  if (!validate_shape(shape, stride)) {
    throw std::invalid_argument(
        "Initial tensor shape and stride must be contiguous!");
  }

  storage = std::make_shared<SparseCpuRealStorage>(val, get_size());

  if (rg) {
    grad->storage->FillZeros();
  }
}
Tensor::Tensor(const ComplexSparseVector &val, std::vector<tcapint> shp,
               std::vector<tcapint> strd, bool rg, bool sg)
    : shape(shp), stride(strd), offset(ZERO_VCI), grad_node(nullptr),
      grad(rg ? std::make_shared<Tensor>(shp, strd, false, DType::COMPLEX,
                                         DeviceTag::CPU, -1, sg)
              : nullptr) {
  if (shape.size() != stride.size()) {
    throw std::invalid_argument(
        "Tensor shape vector must have same length as stride vector!");
  }
  if (!validate_shape(shape, stride)) {
    throw std::invalid_argument(
        "Initial tensor shape and stride must be contiguous!");
  }

  storage = std::make_shared<SparseCpuComplexStorage>(val, get_size());

  if (rg) {
    grad->storage->FillZeros();
  }
}

TensorPtr Tensor::operator[](tcapint idx) {
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
    // during the backward() step.
    TensorPtr g = allocate_like(a, storage->dtype, false, IS_SPARSE(a));
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

    TensorPtr gcp = grad->copy();
    std::vector<tcapint> &sh = gcp->shape;
    std::vector<tcapint> &st = gcp->stride;

    if (sh.size() == 1U) {
      sh[0U] = 1U;
      st[0U] = 0U;
    } else {
      sh.erase(sh.begin() + i);
      st.erase(st.begin() + i);
    }

    TensorPtr tmp =
        allocate_like(gcp, gcp->storage->dtype, false, IS_SPARSE(grad));
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
  if (a->shape.size() > 2U) {
    throw std::invalid_argument("Tensor::transpose is only for 2D tensors (and "
                                "vectors and covectors)!");
  }

  TensorPtr out = a->copy();

  if (out->shape.size() == 1U) {
    // Treat input as column vector, and transpose to row vector
    out->shape = {1U, out->shape[0U]};
    out->stride = {0U, out->stride[0U]};
    return out;
  } else {
    std::swap(out->shape[0], out->shape[1]);
    std::swap(out->stride[0], out->stride[1]);
  }

  return out;
}

TensorPtr Tensor::sum(TensorPtr a) {
  const bool rg = a->requires_grad();
  TensorPtr out = allocate_scalar_like(a, rg);

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
        const DType &dt = out_grad->storage->dtype;
        a_grad->upcast(dt);
        TensorPtr tmp =
            Tensor::allocate_like(out_grad, dt, false, false, false);
        scale->match_shape(out_grad);
        Weed::mul(*(out_grad.get()), *(scale.get()), *(tmp.get()));
        Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
        a->reduce_grad_broadcast();
      });
}

TensorPtr Tensor::mean(TensorPtr a) {
  const bool rg = a->requires_grad();
  TensorPtr out = allocate_scalar_like(a, rg);

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
        scale->match_shape(out_grad);
        TensorPtr tmp =
            Tensor::allocate_like(out_grad, dt, false, false, false);
        Weed::mul(*(out_grad.get()), *(scale.get()), *(tmp.get()));
        Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
        a->reduce_grad_broadcast();
      });
}

TensorPtr Tensor::abs(TensorPtr a) {
  const bool rg = a->requires_grad();
  TensorPtr out =
      allocate_like(a, DType::REAL, rg, IS_SPARSE(a), IS_SPARSE(a->grad));

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
  TensorPtr out =
      allocate_like(a, a->storage->dtype, rg, IS_SPARSE(a), IS_SPARSE(a->grad));

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
        Weed::sigmoid_grad(a_grad, *(out.get()), out_grad);
      });
}

TensorPtr Tensor::relu(TensorPtr a) {
  const bool rg = a->requires_grad();
  TensorPtr out =
      allocate_like(a, a->storage->dtype, rg, IS_SPARSE(a), IS_SPARSE(a->grad));

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
  TensorPtr out =
      allocate_like(a, a->storage->dtype, rg, IS_SPARSE(a), IS_SPARSE(a->grad));

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
  const bool s = IS_SPARSE(a) && IS_SPARSE(b);
  const bool sg = IS_SPARSE(a->grad) && IS_SPARSE(b->grad);
  DType dt = get_dtype_by_presidence({a, b});
  TensorPtr out;
  if (a->get_size() == ONE_VCI) {
    a->match_shape(b);
    out = allocate_like(b, dt, rg, s, sg);
  } else if (b->get_size() == ONE_VCI) {
    b->match_shape(a);
    out = allocate_like(a, dt, rg, s, sg);
  } else {
    out = allocate_like(a, dt, rg, s, sg);
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
  const bool s = IS_SPARSE(a) && IS_SPARSE(b);
  const bool sg = IS_SPARSE(a->grad) && IS_SPARSE(b->grad);
  DType dt = get_dtype_by_presidence({a, b});
  TensorPtr out;
  if (a->get_size() == ONE_VCI) {
    a->match_shape(b);
    out = allocate_like(b, dt, rg, s, sg);
  } else if (b->get_size() == ONE_VCI) {
    b->match_shape(a);
    out = allocate_like(a, dt, rg, s, sg);
  } else {
    out = allocate_like(a, dt, rg, s, sg);
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
          const DType &dt = get_dtype_by_presidence({b, out_grad});
          TensorPtr tmp =
              Tensor::allocate_like(a_grad, dt, false, IS_SPARSE(b));
          Weed::mul(*(out_grad.get()), *(b.get()), *(tmp.get()));
          a_grad->upcast(dt);
          Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
          a->reduce_grad_broadcast();
        }
        if (b->requires_grad()) {
          b->match_shape(a);
          TensorPtr b_grad = b->grad;
          const DType &dt = get_dtype_by_presidence({a, out_grad});
          TensorPtr tmp =
              Tensor::allocate_like(b_grad, dt, false, IS_SPARSE(a));
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

  if (a->shape.size() != 2U) {
    throw std::invalid_argument(
        "Tensor::matmul is only for matrices with 2 indices!");
  }

  const tcapint as0 = a->shape[0U];
  const tcapint bs1 = (b->shape.size() > 1U) ? b->shape[1U] : 1U;
  const std::vector<tcapint> shp = {as0, bs1};
  const std::vector<tcapint> str = {1U, as0};
  const bool rg = a->requires_grad() || b->requires_grad();
  const bool s = IS_SPARSE(a) && IS_SPARSE(b);
  const bool sg = IS_SPARSE(a->grad) && IS_SPARSE(b->grad);
  DType dt = get_dtype_by_presidence({a, b});
  TensorPtr out = allocate_like(shp, str, a, dt, rg, s, sg);

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
          const DType &dt = get_dtype_by_presidence({b, out_grad});
          TensorPtr bt = transpose(b);
          const tcapint ogs = out->grad->shape[0U];
          const std::vector<tcapint> shp = {ogs, bt->shape[1U]};
          const std::vector<tcapint> str = {1U, ogs};
          TensorPtr tmp =
              Tensor::allocate_like(shp, str, a_grad, dt, false, IS_SPARSE(a));
          Weed::matmul(*(out->grad.get()), *(bt.get()), *(tmp.get()));
          a_grad->upcast(dt);
          Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
        }
        if (b->requires_grad()) {
          TensorPtr b_grad = b->grad;
          const DType &dt = get_dtype_by_presidence({a, out_grad});
          TensorPtr at = transpose(a);
          const tcapint ogs =
              (out->grad->shape.size() > 1U) ? out->grad->shape[1U] : 1U;
          const tcapint ats = at->shape[0U];
          const std::vector<tcapint> shp = {ats, ogs};
          const std::vector<tcapint> str = {1U, ats};
          TensorPtr tmp =
              Tensor::allocate_like(shp, str, b_grad, dt, false, IS_SPARSE(a));
          Weed::matmul(*(at.get()), *(out->grad.get()), *(tmp.get()));
          b_grad->upcast(dt);
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
  const bool s = IS_SPARSE(a) && IS_SPARSE(b);
  const bool sg = IS_SPARSE(a->grad) && IS_SPARSE(b->grad);
  DType dt = get_dtype_by_presidence({a, b});
  TensorPtr out;
  if (a->get_size() == ONE_VCI) {
    a->match_shape(b);
    out = allocate_like(b, dt, rg, s, sg);
  } else if (b->get_size() == ONE_VCI) {
    b->match_shape(a);
    out = allocate_like(a, dt, rg, s, sg);
  } else {
    out = allocate_like(a, dt, rg, s, sg);
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
  DType dt = get_dtype_by_presidence({a, b});
  const bool s = IS_SPARSE(a) && IS_SPARSE(b);
  const bool sg = IS_SPARSE(a->grad) && IS_SPARSE(b->grad);
  TensorPtr out;
  if (a->get_size() == ONE_VCI) {
    a->match_shape(b);
    out = allocate_like(b, dt, rg, s, sg);
  } else if (b->get_size() == ONE_VCI) {
    b->match_shape(a);
    out = allocate_like(a, dt, rg, s, sg);
  } else {
    out = allocate_like(a, dt, rg, s, sg);
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
          const DType &dt = get_dtype_by_presidence({b, out_grad});
          a_grad->upcast(dt);
          TensorPtr tmp = Tensor::allocate_like(b, dt, false, IS_SPARSE(b));
          Weed::div(*(out_grad.get()), *(b.get()), *(tmp.get()));
          Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
          a->reduce_grad_broadcast();
        }
        if (b->requires_grad()) {
          b->match_shape(a);
          TensorPtr b_grad = b->grad;
          TensorPtr b_sqr =
              Tensor::allocate_like(b, b->storage->dtype, false, IS_SPARSE(b));
          Weed::mul(*(b.get()), *(b.get()), *(b_sqr.get()));
          const DType &dt = get_dtype_by_presidence({a, b_sqr});
          b_grad->upcast(dt);
          TensorPtr tmp = Tensor::allocate_like(a, dt, false, IS_SPARSE(a));
          Weed::div(*(a.get()), *(b_sqr.get()), *(tmp.get()));
          Weed::sub_in_place(*(b_grad.get()), *(tmp.get()));
          b->reduce_grad_broadcast();
        }
      });
}

TensorPtr Tensor::pow(TensorPtr a, real1 p) {
  const bool rg = a->requires_grad();
  TensorPtr out =
      allocate_like(a, a->storage->dtype, rg, IS_SPARSE(a), IS_SPARSE(a->grad));

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

    TensorPtr dy_y =
        Tensor::allocate_like(dy, dy->storage->dtype, false, IS_SPARSE(dy));
    Weed::mul(*(dy.get()), *(y.get()), *(dy_y.get()));

    TensorPtr dy_y_p = Tensor::allocate_like(dy_y, dy_y->storage->dtype, false,
                                             IS_SPARSE(dy_y));
    p->match_shape(dy_y);
    Weed::mul(*(dy_y.get()), *(p.get()), *(dy_y_p.get()));

    TensorPtr r = Tensor::allocate_like(dy_y_p, dy_y_p->storage->dtype, false,
                                        IS_SPARSE(dy_y_p));
    Weed::div(*(dy_y_p.get()), *(x.get()), *(r.get()));

    dx->upcast(r->storage->dtype);
    Weed::add_in_place(*(dx.get()), *(r.get()));
  });
}

TensorPtr Tensor::exp(TensorPtr a, real1 b) {
  const bool rg = a->requires_grad();
  TensorPtr out =
      allocate_like(a, a->storage->dtype, rg, IS_SPARSE(a), IS_SPARSE(a->grad));

  Weed::exp(*(a.get()), b, *(out.get()));

  if (rg) {
    RealScalarPtr y = std::make_shared<RealScalar>((real1)std::log((real1_s)b),
                                                   false, a->storage->device,
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

        TensorPtr dy_v =
            Tensor::allocate_like(dy, dy->storage->dtype, false, IS_SPARSE(dy));
        log_b->match_shape(dy);
        Weed::mul(*(dy.get()), *(log_b.get()), *(dy_v.get()));

        TensorPtr r = Tensor::allocate_like(dy_v, dy_v->storage->dtype, false,
                                            IS_SPARSE(dy_v));
        Weed::mul(*(dy_v.get()), *(y.get()), *(r.get()));

        dx->upcast(r->storage->dtype);
        Weed::add_in_place(*(dx.get()), *(r.get()));
      });
}

TensorPtr Tensor::log(TensorPtr a, real1 b) {
  const bool rg = a->requires_grad();
  TensorPtr out =
      allocate_like(a, a->storage->dtype, rg, IS_SPARSE(a), IS_SPARSE(a->grad));

  Weed::log(*(a.get()), b, *(out.get()));

  if (rg) {
    RealScalarPtr y = std::make_shared<RealScalar>(
        (real1)(ONE_R1 / std::log((real1_s)b)), false, a->storage->device,
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

        TensorPtr dy_v =
            Tensor::allocate_like(dy, dy->storage->dtype, false, IS_SPARSE(dy));
        inv_log_b->match_shape(dy);
        Weed::mul(*(dy.get()), *(inv_log_b.get()), *(dy_v.get()));

        TensorPtr r = Tensor::allocate_like(dy_v, dy_v->storage->dtype, false,
                                            IS_SPARSE(dy_v));
        Weed::div(*(dy_v.get()), *(y.get()), *(r.get()));

        dx->upcast(r->storage->dtype);
        Weed::add_in_place(*(dx.get()), *(r.get()));
      });
}
} // namespace Weed
