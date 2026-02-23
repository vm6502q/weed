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
#include "ops/copy_broadcast.hpp"
#include "ops/div.hpp"
#include "ops/in_place.hpp"
#include "ops/matmul.hpp"
#include "ops/pow.hpp"
#include "ops/real_extremum.hpp"
#include "ops/real_unary.hpp"
#include "ops/reduce.hpp"
#include "ops/sub.hpp"
#include "ops/sum.hpp"
#include "tensors/real_scalar.hpp"

#include "storage/all_storage.hpp"

#include <thread>
#include <unordered_set>

#define GET_REAL(ptr) static_cast<RealScalar *>((ptr).get())->get_item()
#define IS_SPARSE(a)                                                           \
  (a && a->storage->is_sparse() &&                                             \
   ((a->storage->get_sparse_size() << 1U) < a->storage->size))

#if ENABLE_GPU
#define INIT_DEVICE_STORAGE(val, GpuType, CpuType)                             \
  switch (dtag) {                                                              \
  case DeviceTag::GPU:                                                         \
    storage = std::make_shared<GpuType>(val, did);                             \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    storage = std::make_shared<CpuType>(val);                                  \
  }
#else
// There is only one device type available:
#define INIT_DEVICE_STORAGE(val, GpuType, CpuType)                             \
  storage = std::make_shared<CpuType>(val);
#endif

namespace Weed {
#if ENABLE_GPU
#if WEED_ENABLE_ENV_VARS
const tlenint PSTRIDEPOW_DEFAULT =
    (tlenint)(getenv("WEED_PSTRIDEPOW")
                  ? std::stoi(std::string(getenv("WEED_PSTRIDEPOW")))
                  : WEED_PSTRIDEPOW);
const tcapint GSTRIDE =
    (tcapint)(getenv("WEED_GSTRIDE")
                  ? std::stoi(std::string(getenv("WEED_GSTRIDE")))
                  : ((1 << PSTRIDEPOW_DEFAULT) *
                     std::thread::hardware_concurrency()));
#else
const tlenint PSTRIDEPOW_DEFAULT = WEED_PSTRIDEPOW;
const tcapint GSTRIDE =
    (1 << PSTRIDEPOW_DEFAULT) * std::thread::hardware_concurrency();
#endif
#else
// Never auto-switch to a GPU if we don't have one.
const tcapint GSTRIDE = -1;
#endif

DeviceTag Tensor::get_dtag_by_presidence(const std::vector<TensorPtr> &v) {
#if ENABLE_GPU
  for (const TensorPtr &p : v) {
    const tcapint sz = p->storage->size;
    const tcapint sp = p->storage->get_sparse_size();
    if (sz == sp) {
      if (sz > GSTRIDE) {
        return DeviceTag::GPU;
      }
    } else {
      if ((sp << 1U) > GSTRIDE) {
        return DeviceTag::GPU;
      }
    }
  }
#endif

  return DeviceTag::CPU;
}

void Tensor::copy(const TensorPtr &cp) {
  // A tensor is a view on storage:
  BaseTensor::copy(cp);
  freeze = cp->freeze;
  requires_grad = cp->requires_grad;

  if (!requires_grad) {
    return;
  }

  make_gradient();
  grad_node = std::make_shared<Node>(std::vector<TensorPtr>{cp}, [cp, this]() {
    const DeviceTag dtag = get_dtag_by_presidence({grad, cp->grad});

    TensorPtr dx = cp->grad->cast(dtag);
    TensorPtr dy = grad->cast(dtag);

    dx->match_shape(dy);
    dx->materialize_broadcast();

    Weed::add_in_place(*(dx.get()), *(dy.get()));

    cp->grad = dx;
    cp->reduce_grad_broadcast();
  });
}

void Tensor::make_gradient(const bool &force_sparse) {
  if (!requires_grad || grad) {
    return;
  }

  const tcapint sz = storage->size;
  const tcapint sp = storage->get_sparse_size();
  DeviceTag dtag = DeviceTag::CPU;
  if (sz == sp) {
    if (sz > GSTRIDE) {
      dtag = DeviceTag::GPU;
    }
  } else {
    if ((sp << 1U) > GSTRIDE) {
      dtag = DeviceTag::GPU;
    }
  }

  grad = Tensor::make_gradient(shape, force_sparse || storage->is_sparse(),
                               storage->dtype, dtag, storage->get_device_id());
}

TensorPtr Tensor::allocate_scalar_like(const Tensor &orig, const bool &rg) {
  return allocate_like(std::vector<tcapint>{1U}, std::vector<tcapint>{0U}, orig,
                       orig.storage->dtype, rg, false);
}

TensorPtr Tensor::allocate_like(const Tensor &orig, const DType &dt,
                                const bool &rg, const bool &s) {
  TensorPtr n = allocate_like(orig.shape, full_contiguous_stride(orig.shape),
                              orig, dt, rg, s);
  n->freeze = orig.freeze;

  return n;
}

TensorPtr Tensor::allocate_like(const std::vector<tcapint> &shp,
                                const Tensor &orig, const DType &dt,
                                const bool &rg, const bool &s) {
  return allocate_like(shp, full_contiguous_stride(shp), orig, dt, rg, s);
}

TensorPtr Tensor::allocate_like(const std::vector<tcapint> &shp,
                                const std::vector<tcapint> &strd,
                                const Tensor &orig, const DType &dt,
                                const bool &rg, const bool &s) {
  const DeviceTag dtag = orig.storage->device;
  const int64_t did = orig.storage->get_device_id();

  return std::make_shared<Tensor>(shp, strd, rg, s, dt, dtag, did);
}

Tensor::Tensor(const std::vector<tcapint> &shp,
               const std::vector<tcapint> &strd, const bool &rg, const bool &s,
               const DType &dtype, const DeviceTag &_dtag, const int64_t &did)
    : BaseTensor(shp, strd), grad_node(nullptr), requires_grad(rg),
      freeze(shp.size(), false) {

  validate_dtype(dtype);
  freeze_init_broadcast();

  const tcapint size = get_size();
  DeviceTag dtag = _dtag;
  if (dtag == DEFAULT_DEVICE) {
    if (size > GSTRIDE) {
      dtag = DeviceTag::GPU;
    } else {
      dtag = DeviceTag::CPU;
    }
  }

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
}

Tensor::Tensor(const std::vector<real1> &val, const std::vector<tcapint> &shp,
               const std::vector<tcapint> &strd, const bool &rg,
               const DeviceTag &_dtag, const int64_t &did)
    : BaseTensor(shp, strd), grad_node(nullptr), requires_grad(rg),
      freeze(shp.size(), false) {

  const tcapint size = get_size();

  if (size != val.size()) {
    throw std::invalid_argument("Tensor value initializer vector must have "
                                "same size as implied by shape and stride!");
  }

  freeze_init_broadcast();

  DeviceTag dtag = _dtag;
  if (dtag == DEFAULT_DEVICE) {
    if (size > GSTRIDE) {
      dtag = DeviceTag::GPU;
    } else {
      dtag = DeviceTag::CPU;
    }
  }

#if ENABLE_GPU
  INIT_DEVICE_STORAGE(val, GpuRealStorage, CpuRealStorage);
#else
  storage = std::make_shared<CpuRealStorage>(val);
#endif
}

Tensor::Tensor(const std::vector<complex> &val, const std::vector<tcapint> &shp,
               const std::vector<tcapint> &strd, const bool &rg,
               const DeviceTag &_dtag, const int64_t &did)
    : BaseTensor(shp, strd), grad_node(nullptr), requires_grad(rg),
      freeze(shp.size(), false) {

  const tcapint size = get_size();

  if (size != val.size()) {
    throw std::invalid_argument("Tensor value initializer vector must have "
                                "same size as implied by shape and stride!");
  }

  freeze_init_broadcast();

  DeviceTag dtag = _dtag;
  if (dtag == DEFAULT_DEVICE) {
    if (size > GSTRIDE) {
      dtag = DeviceTag::GPU;
    } else {
      dtag = DeviceTag::CPU;
    }
  }

#if ENABLE_GPU
  INIT_DEVICE_STORAGE(val, GpuComplexStorage, CpuComplexStorage);
#else
  storage = std::make_shared<CpuComplexStorage>(val);
#endif
}

Tensor::Tensor(const RealSparseVector &val, const std::vector<tcapint> &shp,
               const std::vector<tcapint> &strd, const bool &rg)
    : BaseTensor(shp, strd), grad_node(nullptr), requires_grad(rg),
      freeze(shp.size(), false) {
  freeze_init_broadcast();
  storage = std::make_shared<SparseCpuRealStorage>(val, get_size());
}
Tensor::Tensor(const ComplexSparseVector &val, const std::vector<tcapint> &shp,
               const std::vector<tcapint> &strd, const bool &rg)
    : BaseTensor(shp, strd), grad_node(nullptr), requires_grad(rg),
      freeze(shp.size(), false) {
  freeze_init_broadcast();
  storage = std::make_shared<SparseCpuComplexStorage>(val, get_size());
}

TensorPtr Tensor::operator[](const tcapint &idx) const {
  if (idx > shape.back()) {
    throw std::invalid_argument("Tensor index out-of-range!");
  }

  TensorPtr v = std::make_shared<Tensor>(*this);
  v->offset += idx * stride.back();
  v->shape.pop_back();
  v->stride.pop_back();

  if (v->shape.empty()) {
    v->shape = {1U};
    v->stride = {0U};
  }

  return v;
}

std::vector<TensorPtr> filterParents(const std::vector<TensorPtr> &parents) {
  std::vector<TensorPtr> filtered;
  for (TensorPtr p : parents) {
    if (p->requires_grad) {
      filtered.push_back(p);
    }
  }

  return filtered;
}

bool Tensor::match_shape(const TensorPtr a) {
  if (shape.size() > a->shape.size()) {
    return false;
  }

  std::vector<tcapint> osh = shape;
  std::vector<tcapint> ost = stride;
  std::vector<tcapint> nsh = a->shape;

  std::reverse(osh.begin(), osh.end());
  std::reverse(ost.begin(), ost.end());
  std::reverse(nsh.begin(), nsh.end());

  for (size_t i = 0U; i < osh.size(); ++i) {
    if ((osh[i] != nsh[i]) && ost[i]) {
      return false;
    }
  }

  std::vector<bool> frz = freeze;
  std::reverse(frz.begin(), frz.end());

  ost.resize(nsh.size());
  frz.resize(nsh.size());

  std::reverse(ost.begin(), ost.end());
  std::reverse(frz.begin(), frz.end());

  shape = a->shape;
  stride = ost;
  freeze = frz;

  return true;
}

void Tensor::materialize_broadcast() {
  bool needs = false;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] > 1U && stride[i] == 0) {
      needs = true;
      break;
    }
  }
  if (!needs) {
    return;
  }

  TensorPtr tmp = Tensor::allocate_like(shape, *this, storage->dtype,
                                        requires_grad, storage->is_sparse());

  // copy via kernel (broadcast-aware read)
  Weed::copy_broadcast(*tmp, *this);

  *this = *tmp;
}

void Tensor::reduce_grad_broadcast() {
  if (!requires_grad || !grad) {
    throw std::domain_error(
        "Called Tensor::reduce_grad_broadcast() on a node instance without a "
        "gradient Tensor! (This should be called only during autograd.)");
  }

  for (symint i = stride.size() - 1; i >= 0; --i) {
    if (freeze[i] || stride[i]) {
      continue;
    }

    grad = sum(grad, i);
  }
}

void Tensor::backward(TensorPtr loss) {
  if (!loss || !loss->requires_grad) {
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

TensorPtr Tensor::softmax(const TensorPtr x, symint axis) {
  while (axis < 0) {
    axis += x->shape.size();
  }

  TensorPtr m = max(x, axis);
  TensorPtr x_shifted = x - m;
  TensorPtr ex = exp(x_shifted);
  TensorPtr denom = sum(ex, axis);

  return ex / denom;
}

TensorPtr Tensor::logsoftmax(const TensorPtr x, symint axis) {
  while (axis < 0) {
    axis += x->shape.size();
  }

  TensorPtr m = Tensor::max(x, axis);
  TensorPtr x_shifted = x - m;
  TensorPtr logsum = Tensor::log(Tensor::sum(Tensor::exp(x_shifted), axis));

  return x_shifted - logsum;
}

TensorPtr Tensor::slice(TensorPtr a, const int64_t &row) {
  const bool rg = a->requires_grad;

  TensorPtr out = std::make_shared<Tensor>(*(a.get()));

  out->offset += row * a->stride[0U];
  out->shape[0U] = 1U;

  // Remove dimension BEFORE making grad node
  out->shape.erase(out->shape.begin());
  out->stride.erase(out->stride.begin());

  if (rg) {
    make_row_slice_node(a, out, row);
  }

  return out;
}

void Tensor::make_row_slice_node(TensorPtr a, TensorPtr out,
                                 const tcapint &row) {
  out->make_gradient();
  out->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a}, [a, out, row]() {
        const DeviceTag dtag = get_dtag_by_presidence({a, a->grad, out->grad});

        TensorPtr a_grad = a->grad->cast(dtag);
        TensorPtr out_grad = out->grad->cast(dtag);

        a_grad->materialize_broadcast();

        TensorPtr row_view = Tensor::slice(a_grad, row);

        Weed::add_in_place(*(row_view.get()), *(out_grad.get()));

        a->grad = a_grad;
        a->reduce_grad_broadcast();
      });
}

// Contributed by Elara (the OpenAI custom GPT)
TensorPtr Tensor::slice(TensorPtr a, int64_t axis, const tcapint &start,
                        const tcapint &length) {

  while (axis < 0) {
    axis += a->shape.size();
  }

  if (axis >= (int64_t)a->shape.size()) {
    throw std::invalid_argument("Tensor::slice: axis out of range");
  }

  if (start < 0 || length <= 0 || start + length > a->shape[axis]) {
    throw std::invalid_argument("Tensor::slice: invalid range");
  }

  const bool rg = a->requires_grad;
  TensorPtr out = std::make_shared<Tensor>(*(a.get()));

  out->offset += start * a->stride[axis];
  out->shape[axis] = length;

  if (rg) {
    make_slice_node(a, out, axis, start);
  }

  return out;
}

void Tensor::make_slice_node(TensorPtr a, TensorPtr out, const int64_t &axis,
                             const tcapint &start) {
  out->make_gradient();
  out->grad_node = std::make_shared<Node>(
      std::vector<TensorPtr>{a}, [a, out, axis, start]() {
        const DeviceTag dtag = get_dtag_by_presidence({a, a->grad, out->grad});

        TensorPtr a_grad = a->grad->cast(dtag);
        TensorPtr out_grad = out->grad->cast(dtag);

        a_grad->materialize_broadcast();

        // Create zero tensor same shape as parent
        TensorPtr tmp = allocate_like(*(a_grad.get()), a_grad->storage->dtype,
                                      false, IS_SPARSE(out_grad));

        // Create view into tmp matching slice region
        TensorPtr tmp_slice = slice(tmp, axis, start, out_grad->shape[axis]);

        // Copy gradient into correct region
        Weed::add_in_place(*(tmp_slice.get()), *(out_grad.get()));

        Weed::add_in_place(*(a_grad.get()), *(tmp.get()));

        a->grad = a_grad;
        a->reduce_grad_broadcast();
      });
}

TensorPtr Tensor::sum(TensorPtr a) {
  const bool rg = a->requires_grad;
  TensorPtr out = allocate_scalar_like(*(a.get()), rg);

  Weed::sum(*(a.get()), *(out.get()));

  if (rg) {
    make_sum_node(a, out);
  }

  return out;
}

void Tensor::make_sum_node(TensorPtr a, TensorPtr out) {
  out->make_gradient();
  out->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a}, [a, out]() {
        const DeviceTag dtag = get_dtag_by_presidence({a->grad, out->grad});
        TensorPtr a_grad = a->grad->cast(dtag);
        TensorPtr out_grad = out->grad->cast(dtag);
        // da += dout  (broadcast)
        out_grad->match_shape(a_grad);
        a_grad->upcast(out_grad->storage->dtype);
        a_grad->match_shape(out_grad);
        a_grad->materialize_broadcast();
        Weed::add_in_place(*(a_grad.get()), *(out_grad.get()));
        a->grad = a_grad;
        a->reduce_grad_broadcast();
      });
}

TensorPtr Tensor::mean(TensorPtr a) {
  const bool rg = a->requires_grad;
  TensorPtr out = allocate_scalar_like(*(a.get()), rg);

  Weed::mean(*(a.get()), *(out.get()));

  if (rg) {
    make_mean_node(a, out);
  }

  return out;
}

void Tensor::make_mean_node(TensorPtr a, TensorPtr out) {
  out->make_gradient();
  out->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a}, [a, out]() {
        const DeviceTag dtag = get_dtag_by_presidence({a->grad, out->grad});
        TensorPtr a_grad = a->grad->cast(dtag);
        TensorPtr out_grad = out->grad->cast(dtag);
        // da += dout / N   (broadcast)
        a_grad->upcast(out_grad->storage->dtype);
        out_grad->match_shape(a_grad);
        a_grad->match_shape(out_grad);
        a_grad->materialize_broadcast();
        out_grad->match_shape(a_grad);
        TensorPtr s =
            SCALAR((real1)(ONE_R1 / (real1)a->get_broadcast_size()), out_grad);
        TensorPtr tmp = s * out_grad;
        Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
        a->grad = a_grad;
        a->reduce_grad_broadcast();
      });
}

TensorPtr Tensor::sum(TensorPtr a, symint axis) {
  while (axis < 0) {
    axis += a->shape.size();
  }

  const size_t p_stride = a->stride[axis];
  if (!p_stride) {
    a->shape[axis] = 1U;

    return a;
  }

  a = contiguous(a);

  const bool rg = a->requires_grad;
  TensorPtr acp = std::make_shared<Tensor>(*(a.get()));
  acp->shape[axis] = 1U;
  std::vector<tcapint> &st = acp->stride;
  st[axis] = 0U;
  size_t j = axis + 1;
  while (!st[j] && (j < st.size())) {
    ++j;
  }
  if (j < st.size()) {
    const size_t o_stride = st[j] / p_stride;
    for (; j < st.size(); ++j) {
      st[j] /= o_stride;
    }
  }

  TensorPtr out =
      allocate_like(*(acp.get()), acp->storage->dtype, rg, IS_SPARSE(a));
  Weed::reduce(axis, *(a.get()), *(out.get()));

  if (rg) {
    make_sum_node(a, out, axis);
  }

  return out;
}

void Tensor::make_sum_node(TensorPtr a, TensorPtr out, const tcapint &axis) {
  out->make_gradient();
  out->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a}, [a, out, axis]() {
        const DeviceTag dtag = get_dtag_by_presidence({a->grad, out->grad});

        TensorPtr dx = a->grad->cast(dtag);
        TensorPtr dy = std::make_shared<Tensor>(*(out->grad.get()))->cast(dtag);

        // re-insert reduced axis
        dy->shape[axis] = a->shape[axis];
        dx->match_shape(dy);
        dx->materialize_broadcast();

        dx->upcast(dy->storage->dtype);
        Weed::reduce_grad(axis, *dx, *a, *dy);

        a->grad = dx;
        a->reduce_grad_broadcast();
      });
}

TensorPtr Tensor::mean(TensorPtr a, symint axis) {
  while (axis < 0) {
    axis += a->shape.size();
  }

  TensorPtr tmp = sum(a, axis);
  tmp->squeeze(axis);
  tmp = tmp / (real1)(a->shape[axis]);
  tmp->unsqueeze(axis);

  return tmp;
}

TensorPtr Tensor::variance(TensorPtr a) {
  return ((a - mean(a)) ^ real1(2)) / (real1)(a->get_broadcast_size());
}

TensorPtr Tensor::variance(TensorPtr a, const tcapint &axis) {
  TensorPtr tmp = mean(a, axis);
  tmp = a - tmp;

  return mean(tmp * tmp, axis);
}

TensorPtr Tensor::max(TensorPtr a, symint axis) {
  while (axis < 0) {
    axis += a->shape.size();
  }

  const size_t p_stride = a->stride[axis];
  if (!p_stride) {
    a->shape[axis] = 1U;

    return a;
  }

  a = contiguous(a);

  const bool rg = a->requires_grad;
  TensorPtr acp = std::make_shared<Tensor>(*(a.get()));
  acp->shape[axis] = 1U;
  std::vector<tcapint> &st = acp->stride;
  st[axis] = 0U;
  size_t j = axis + 1;
  while (!st[j] && (j < st.size())) {
    ++j;
  }
  if (j < st.size()) {
    const size_t o_stride = st[j] / p_stride;
    for (; j < st.size(); ++j) {
      st[j] /= o_stride;
    }
  }

  TensorPtr out =
      allocate_like(*(acp.get()), acp->storage->dtype, rg, IS_SPARSE(a));
  Weed::max(axis, *(a.get()), *(out.get()));

  if (rg) {
    make_match_node(a, out, axis);
  }

  return out;
}

TensorPtr Tensor::min(TensorPtr a, symint axis) {
  while (axis < 0) {
    axis += a->shape.size();
  }

  const size_t p_stride = a->stride[axis];
  if (!p_stride) {
    a->shape[axis] = 1U;

    return a;
  }

  a = contiguous(a);

  const bool rg = a->requires_grad;
  TensorPtr acp = std::make_shared<Tensor>(*(a.get()));
  acp->shape[axis] = 1U;
  std::vector<tcapint> &st = acp->stride;
  size_t j = axis + 1;
  while (!st[j] && (j < st.size())) {
    ++j;
  }
  if (j < st.size()) {
    const size_t o_stride = st[j] / p_stride;
    for (; j < st.size(); ++j) {
      st[j] /= o_stride;
    }
  }

  TensorPtr out =
      allocate_like(*(acp.get()), acp->storage->dtype, rg, IS_SPARSE(a));
  Weed::min(axis, *(a.get()), *(out.get()));

  if (rg) {
    make_match_node(a, out, axis);
  }

  return out;
}

void Tensor::make_match_node(TensorPtr a, TensorPtr out, const tcapint &axis) {
  out->make_gradient();
  out->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a}, [a, out, axis]() {
        const DeviceTag dtag = get_dtag_by_presidence({a->grad, out->grad});

        TensorPtr dx = a->grad->cast(dtag);
        TensorPtr dy = std::make_shared<Tensor>(*(out->grad.get()))->cast(dtag);

        // re-insert reduced axis
        dy->shape[axis] = a->shape[axis];
        dx->match_shape(dy);
        dx->materialize_broadcast();

        dx->upcast(dy->storage->dtype);
        Weed::match_grad(axis, *dx, *a, *dy, *out);

        a->grad = dx;
        a->reduce_grad_broadcast();
      });
}

TensorPtr Tensor::abs(TensorPtr a) {
  const bool rg = a->requires_grad;
  TensorPtr out = allocate_like(*(a.get()), DType::REAL, rg, IS_SPARSE(a));

  Weed::abs(*(a.get()), *(out.get()));

  if (rg) {
    make_abs_node(a, out);
  }

  return out;
}

void Tensor::make_abs_node(TensorPtr a, TensorPtr out) {
  out->make_gradient();
  out->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a}, [a, out]() {
        const DeviceTag dtag = get_dtag_by_presidence({a, a->grad, out->grad});
        TensorPtr _a = a->cast(dtag);
        TensorPtr a_grad = a->grad->cast(dtag);
        TensorPtr out_grad = out->grad->cast(dtag);
        a_grad->upcast(out_grad->storage->dtype);
        a_grad->match_shape(out_grad);
        a_grad->materialize_broadcast();
        Weed::abs_grad(*(a_grad.get()), *(_a.get()), *(out_grad.get()));
        a->grad = a_grad;
        a->reduce_grad_broadcast();
      });
}

TensorPtr Tensor::gelu(const TensorPtr x) {
  const real1 k0 = real1(0.5);
  const real1 k1 = real1(0.044715);
  const real1 k2 = real1(0.7978845608028654); // sqrt(2/pi)

  TensorPtr x3 = x * x * x;
  TensorPtr inner = k2 * (x + k1 * x3);
  TensorPtr t = Tensor::tanh(inner);

  return k0 * x * (Tensor::ones_like(x->shape) + t);
}

TensorPtr Tensor::relu(TensorPtr a) {
  const bool rg = a->requires_grad;
  TensorPtr out =
      allocate_like(*(a.get()), a->storage->dtype, rg, IS_SPARSE(a));

  Weed::relu(*(a.get()), *(out.get()));

  if (rg) {
    make_relu_node(a, out);
  }

  return out;
}

void Tensor::make_relu_node(TensorPtr a, TensorPtr out) {
  out->make_gradient();
  out->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a}, [a, out]() {
        const DeviceTag dtag = get_dtag_by_presidence({a, a->grad, out->grad});
        TensorPtr _a = a->cast(dtag);
        TensorPtr a_grad = a->grad->cast(dtag);
        TensorPtr out_grad = out->grad->cast(dtag);
        a_grad->upcast(out_grad->storage->dtype);
        a_grad->match_shape(out_grad);
        a_grad->materialize_broadcast();
        Weed::relu_grad(*(a_grad.get()), *(_a.get()), *(out_grad.get()));
        a->grad = a_grad;
        a->reduce_grad_broadcast();
      });
}

TensorPtr Tensor::sigmoid(TensorPtr a) {
  const bool rg = a->requires_grad;
  TensorPtr out =
      allocate_like(*(a.get()), a->storage->dtype, rg, IS_SPARSE(a));

  Weed::sigmoid(*(a.get()), *(out.get()));

  if (rg) {
    make_sigmoid_node(a, out);
  }

  return out;
}

void Tensor::make_sigmoid_node(TensorPtr a, TensorPtr out) {
  out->make_gradient();
  out->grad_node = std::make_shared<Node>(std::vector<TensorPtr>{a}, [a,
                                                                      out]() {
    const DeviceTag dtag = get_dtag_by_presidence({out, a->grad, out->grad});
    TensorPtr a_grad = a->grad->cast(dtag);
    TensorPtr out_grad = out->grad->cast(dtag);
    TensorPtr _out = out->cast(dtag);
    a_grad->upcast(out_grad->storage->dtype);
    a_grad->match_shape(out_grad);
    a_grad->materialize_broadcast();
    Weed::sigmoid_grad(*(a_grad.get()), *(_out.get()), *(out_grad.get()));
    a->grad = a_grad;
    a->reduce_grad_broadcast();
  });
}

TensorPtr Tensor::tanh(TensorPtr a) {
  const bool rg = a->requires_grad;
  TensorPtr out =
      allocate_like(*(a.get()), a->storage->dtype, rg, IS_SPARSE(a));

  Weed::tanh(*(a.get()), *(out.get()));

  if (rg) {
    make_tanh_node(a, out);
  }

  return out;
}

void Tensor::make_tanh_node(TensorPtr a, TensorPtr out) {
  out->make_gradient();
  out->grad_node = std::make_shared<Node>(std::vector<TensorPtr>{a}, [a,
                                                                      out]() {
    const DeviceTag dtag = get_dtag_by_presidence({out, a->grad, out->grad});
    TensorPtr a_grad = a->grad->cast(dtag);
    TensorPtr out_grad = out->grad->cast(dtag);
    TensorPtr _out = out->cast(dtag);
    a_grad->upcast(out_grad->storage->dtype);
    a_grad->match_shape(out_grad);
    a_grad->materialize_broadcast();
    Weed::tanh_grad(*(a_grad.get()), *(_out.get()), *(out_grad.get()));
    a->grad = a_grad;
    a->reduce_grad_broadcast();
  });
}

TensorPtr Tensor::max(TensorPtr a) {
  const bool rg = a->requires_grad;
  TensorPtr out = allocate_scalar_like(*(a.get()), rg);

  Weed::max(*(a.get()), *(out.get()));

  if (rg) {
    make_max_node(a, out);
  }

  return out;
}

void Tensor::make_max_node(TensorPtr a, TensorPtr out) {
  out->make_gradient();
  a->make_gradient(true);
  out->grad_node = std::make_shared<Node>(std::vector<TensorPtr>{a}, [a,
                                                                      out]() {
    const DeviceTag dtag = get_dtag_by_presidence({a, out, a->grad, out->grad});
    TensorPtr _a = a->cast(dtag);
    TensorPtr _out = out->cast(dtag);
    TensorPtr a_grad = a->grad->cast(dtag);
    TensorPtr out_grad = out->grad->cast(dtag);
    a_grad->upcast(out_grad->storage->dtype);
    a_grad->match_shape(out_grad);
    a_grad->materialize_broadcast();
    out_grad->match_shape(a_grad);
    Weed::max_grad(*(a_grad.get()), *(_a.get()), *(out_grad.get()),
                   *(_out.get()));
    a->grad = a_grad;
    a->reduce_grad_broadcast();
  });
}

TensorPtr Tensor::min(TensorPtr a) {
  const bool rg = a->requires_grad;
  TensorPtr out = allocate_scalar_like(*(a.get()), rg);

  Weed::min(*(a.get()), *(out.get()));

  if (rg) {
    make_min_node(a, out);
  }

  return out;
}

void Tensor::make_min_node(TensorPtr a, TensorPtr out) {
  out->make_gradient();
  a->make_gradient(true);
  out->grad_node = std::make_shared<Node>(std::vector<TensorPtr>{a}, [a,
                                                                      out]() {
    const DeviceTag dtag = get_dtag_by_presidence({a, out, a->grad, out->grad});
    TensorPtr _a = a->cast(dtag);
    TensorPtr _out = out->cast(dtag);
    TensorPtr a_grad = a->grad->cast(dtag);
    TensorPtr out_grad = out->grad->cast(dtag);
    a_grad->upcast(out_grad->storage->dtype);
    a_grad->match_shape(out_grad);
    a_grad->materialize_broadcast();
    out_grad->match_shape(a_grad);
    Weed::min_grad(*(a_grad.get()), *(_a.get()), *(out_grad.get()),
                   *(_out.get()));
    a->grad = a_grad;
    a->reduce_grad_broadcast();
  });
}

TensorPtr Tensor::clamp(TensorPtr a, real1 lo, real1 hi) {
  const bool rg = a->requires_grad;
  TensorPtr out =
      allocate_like(*(a.get()), a->storage->dtype, rg, IS_SPARSE(a));

  Weed::clamp(*(a.get()), lo, hi, *(out.get()));

  if (rg) {
    make_clamp_node(a, lo, hi, out);
  }

  return out;
}

void Tensor::make_clamp_node(TensorPtr a, real1 lo, real1 hi, TensorPtr out) {
  out->make_gradient();
  out->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a}, [a, out, lo, hi]() {
        const DeviceTag dtag = get_dtag_by_presidence({a, a->grad, out->grad});
        TensorPtr _a = a->cast(dtag);
        TensorPtr a_grad = a->grad->cast(dtag);
        TensorPtr out_grad = out->grad->cast(dtag);
        a_grad->upcast(out_grad->storage->dtype);
        a_grad->match_shape(out_grad);
        a_grad->materialize_broadcast();
        Weed::clamp_grad(*(a_grad.get()), *(_a.get()), *(out_grad.get()), lo,
                         hi);
        a->grad = a_grad;
        a->reduce_grad_broadcast();
      });
}

TensorPtr Tensor::add(TensorPtr a, TensorPtr b) {
  const DeviceTag dtag = get_dtag_by_presidence({a, b});
  a->cast_in_place(dtag);
  b->cast_in_place(dtag);
  const bool rg = a->requires_grad || b->requires_grad;
  const bool s = IS_SPARSE(a) && IS_SPARSE(b);
  const DType dt = get_dtype_by_presidence({a, b});
  if (!a->match_shape(b) && !b->match_shape(a)) {
    throw std::invalid_argument("Tensor::match_shape() failed! (You tried to "
                                "alter an index that was not broadcast.)");
  }
  TensorPtr out = Tensor::allocate_like(a->shape, *(a.get()), dt, rg, s);

  Weed::add(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_add_node(a, b, out);
  }

  return out;
}

void Tensor::make_add_node(TensorPtr a, TensorPtr b, TensorPtr out) {
  out->make_gradient();
  out->grad_node = std::make_shared<Node>(filterParents({a, b}), [a, b, out]() {
    std::vector<TensorPtr> p{out->grad};
    if (a->requires_grad) {
      p.push_back(a->grad);
    }
    if (b->requires_grad) {
      p.push_back(b->grad);
    }
    const DeviceTag dtag = get_dtag_by_presidence(p);
    TensorPtr out_grad = out->grad->cast(dtag);
    if (a->requires_grad) {
      TensorPtr a_grad = a->grad->cast(dtag);
      a_grad->upcast(out_grad->storage->dtype);
      a_grad->match_shape(out_grad);
      a_grad->materialize_broadcast();
      Weed::add_in_place(*(a_grad.get()), *(out_grad.get()));
      a->grad = a_grad;
      a->reduce_grad_broadcast();
    }
    if (b->requires_grad) {
      TensorPtr b_grad = b->grad->cast(dtag);
      b_grad->upcast(out_grad->storage->dtype);
      b_grad->match_shape(out_grad);
      b_grad->materialize_broadcast();
      Weed::add_in_place(*(b_grad.get()), *(out_grad.get()));
      b->grad = b_grad;
      b->reduce_grad_broadcast();
    }
  });
}

TensorPtr Tensor::mul(TensorPtr a, TensorPtr b) {
  const DeviceTag dtag = get_dtag_by_presidence({a, b});
  a->cast_in_place(dtag);
  b->cast_in_place(dtag);
  const bool rg = a->requires_grad || b->requires_grad;
  const bool s = IS_SPARSE(a) && IS_SPARSE(b);
  const DType dt = get_dtype_by_presidence({a, b});
  if (!a->match_shape(b) && !b->match_shape(a)) {
    throw std::invalid_argument("Tensor::match_shape() failed! (You tried to "
                                "alter an index that was not broadcast.)");
  }
  TensorPtr out = Tensor::allocate_like(a->shape, *(a.get()), dt, rg, s);

  Weed::mul(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_mul_node(a, b, out);
  }

  return out;
}

void Tensor::make_mul_node(TensorPtr a, TensorPtr b, TensorPtr out) {
  out->make_gradient();
  out->grad_node = std::make_shared<Node>(filterParents({a, b}), [a, b, out]() {
    std::vector<TensorPtr> p{out->grad};
    if (a->requires_grad) {
      p.push_back(a->grad);
      p.push_back(b);
    }
    if (b->requires_grad) {
      p.push_back(b->grad);
      p.push_back(a);
    }
    const DeviceTag dtag = get_dtag_by_presidence(p);
    TensorPtr out_grad = out->grad->cast(dtag);
    if (a->requires_grad) {
      TensorPtr _b = b->cast(dtag);
      TensorPtr a_grad = a->grad->cast(dtag);
      const DType &dt = get_dtype_by_presidence({_b, out_grad});
      TensorPtr tmp =
          Tensor::allocate_like(*(a_grad.get()), dt, false, IS_SPARSE(b));
      Weed::mul(*(out_grad.get()), *(_b.get()), *(tmp.get()));
      a_grad->upcast(dt);
      a_grad->match_shape(out_grad);
      a_grad->materialize_broadcast();
      Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
      a->grad = a_grad;
      a->reduce_grad_broadcast();
    }
    if (b->requires_grad) {
      TensorPtr _a = a->cast(dtag);
      TensorPtr b_grad = b->grad->cast(dtag);
      const DType &dt = get_dtype_by_presidence({_a, out_grad});
      TensorPtr tmp =
          Tensor::allocate_like(*(b_grad.get()), dt, false, IS_SPARSE(a));
      Weed::mul(*(out_grad.get()), *(_a.get()), *(tmp.get()));
      b_grad->upcast(dt);
      b_grad->match_shape(out_grad);
      b_grad->materialize_broadcast();
      Weed::add_in_place(*(b_grad.get()), *(tmp.get()));
      b->grad = b_grad;
      b->reduce_grad_broadcast();
    }
  });
}

TensorPtr Tensor::matmul(TensorPtr a, TensorPtr b) {
  if (a->shape.size() < 2U) {
    throw std::invalid_argument("Tensor::matmul requires a to have rank >= 2");
  }

  const bool rg = a->requires_grad || b->requires_grad;
  const bool s = IS_SPARSE(a) && IS_SPARSE(b);
  const DType dt = get_dtype_by_presidence({a, b});

  if (a->shape.size() > 2U && b->shape.size() > 2U) {
    if (a->shape.size() != b->shape.size()) {
      throw std::invalid_argument("batched matmul rank mismatch");
    }

    size_t rank = a->shape.size();

    // Check batch dims
    for (size_t i = 0; i < rank - 2; ++i) {
      if (a->shape[i] != b->shape[i]) {
        throw std::invalid_argument("batched matmul batch mismatch");
      }
    }

    symint M = a->shape[rank - 2];
    symint K = a->shape[rank - 1];
    symint K2 = b->shape[rank - 2];
    symint N = b->shape[rank - 1];

    if (K != K2) {
      throw std::invalid_argument("batched matmul inner dim mismatch");
    }

    symint batch = 1;
    for (size_t i = 0; i < rank - 2; ++i) {
      batch *= a->shape[i];
    }

    // reshape to 3D
    TensorPtr a3 = reshape(a, {batch, M, K});
    TensorPtr b3 = reshape(b, {batch, K, N});

    // allocate output
    std::vector<tcapint> out_shape;
    for (size_t i = 0; i < rank - 2; ++i) {
      out_shape.push_back(a->shape[i]);
    }
    out_shape.push_back(M);
    out_shape.push_back(N);

    TensorPtr out = allocate_like(out_shape, full_contiguous_stride(out_shape),
                                  *(a3.get()), dt, rg, s);

    TensorPtr out3 = reshape(out, {batch, M, N});

    // loop over batch
    for (symint i = 0; i < batch; ++i) {
      TensorPtr ai = slice(a3, i);
      TensorPtr bi = slice(b3, i);
      TensorPtr oi = slice(out3, i);

      Weed::matmul(*(ai.get()), *(bi.get()), *(oi.get()));

      if (rg) {
        make_matmul_node(ai, bi, oi);
      }
    }

    return out;
  }

  const bool needs_flatten = (a->shape.size() > 2U);

  const symint K = a->shape.back();
  const symint M = a->shape[a->shape.size() - 2];
  const symint N = b->shape[1U];

  if ((symint)(b->shape[0U]) != K) {
    throw std::invalid_argument("matmul dimension mismatch");
  }

  symint batch = 1;
  for (size_t i = 0; i < a->shape.size() - 2; ++i) {
    batch *= a->shape[i];
  }

  TensorPtr a2 = a;
  if (needs_flatten) {
    a2 = reshape(a, {batch * M, K});
  }

  const DeviceTag dtag = get_dtag_by_presidence({a2, b});
  a2->cast_in_place(dtag);
  b->cast_in_place(dtag);

  const tcapint as0 = a2->shape[0U];
  const tcapint bs1 = b->shape[1U];
  const std::vector<tcapint> shp = {as0, bs1};
  const std::vector<tcapint> str = {1U, as0};
  TensorPtr out = allocate_like(shp, str, *(a2.get()), dt, rg, s);

  Weed::matmul(*(a2.get()), *(b.get()), *(out.get()));

  if (needs_flatten) {
    std::vector<symint> final_shape;
    for (size_t i = 0; i < a->shape.size() - 2; ++i) {
      final_shape.push_back(a->shape[i]);
    }
    final_shape.push_back(M);
    final_shape.push_back(N);

    out = reshape(out, final_shape);
  }

  if (rg) {
    make_matmul_node(a, b, out);
  }

  return out;
}

void Tensor::make_matmul_node(TensorPtr a, TensorPtr b, TensorPtr out) {
  out->make_gradient();
  out->grad_node = std::make_shared<Node>(filterParents({a, b}), [a, b, out]() {
    std::vector<TensorPtr> p{out->grad};
    if (a->requires_grad) {
      p.push_back(a->grad);
      p.push_back(b);
    }
    if (b->requires_grad) {
      p.push_back(b->grad);
      p.push_back(a);
    }
    const DeviceTag dtag = get_dtag_by_presidence(p);
    TensorPtr out_grad = out->grad->cast(dtag);

    const bool needs_flatten = (a->shape.size() > 2U);
    const symint K = a->shape.back();
    const symint M = a->shape[a->shape.size() - 2];
    const symint N = b->shape[1U];

    symint batch = 1;
    for (size_t i = 0; i < a->shape.size() - 2; ++i) {
      batch *= a->shape[i];
    }

    TensorPtr a2 = a;
    TensorPtr out_grad2 = out_grad;

    if (needs_flatten) {
      a2 = reshape(a, {batch * M, K});
      out_grad2 = reshape(out_grad, {batch * M, N});
    }

    if (a->requires_grad) {
      TensorPtr a_grad = a->grad->cast(dtag);
      TensorPtr bt = transpose(b)->cast(dtag);

      const DType &dt = get_dtype_by_presidence({b, out_grad});
      TensorPtr tmp = Tensor::allocate_like(
          std::vector<tcapint>{(tcapint)(batch * M),
                               (tcapint)K}, // 2D flattened shape
          std::vector<tcapint>{1U, (tcapint)(batch * M)}, *(a2.get()), dt,
          false, IS_SPARSE(out_grad));

      Weed::matmul(*(out_grad2.get()), *(bt.get()), *(tmp.get()));

      if (needs_flatten) {
        std::vector<symint> a_shape(a->shape.size());
        for (size_t i = 0U; i < a_shape.size(); ++i) {
          a_shape[i] = (symint)(a->shape[i]);
        }
        tmp = reshape(tmp, a_shape); // restore (..., M, K)
      }

      a_grad->upcast(dt);
      Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
      a->grad = a_grad;
    }

    if (b->requires_grad) {
      TensorPtr b_grad = b->grad->cast(dtag);
      TensorPtr at = transpose(a2)->cast(dtag);

      const DType &dt = get_dtype_by_presidence({a, out_grad});
      TensorPtr tmp = Tensor::allocate_like(*(b_grad.get()), dt, false,
                                            IS_SPARSE(out_grad));

      Weed::matmul(*(at.get()), *(out_grad2.get()), *(tmp.get()));

      b_grad->upcast(dt);
      Weed::add_in_place(*(b_grad.get()), *(tmp.get()));
      b->grad = b_grad;
    }
  });
}

TensorPtr Tensor::sub(TensorPtr a, TensorPtr b) {
  const DeviceTag dtag = get_dtag_by_presidence({a, b});
  a->cast_in_place(dtag);
  b->cast_in_place(dtag);
  const bool rg = a->requires_grad || b->requires_grad;
  const bool s = IS_SPARSE(a) && IS_SPARSE(b);
  const DType dt = get_dtype_by_presidence({a, b});
  if (!a->match_shape(b) && !b->match_shape(a)) {
    throw std::invalid_argument("Tensor::match_shape() failed! (You tried to "
                                "alter an index that was not broadcast.)");
  }
  TensorPtr out = Tensor::allocate_like(a->shape, *(a.get()), dt, rg, s);

  Weed::sub(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_sub_node(a, b, out);
  }

  return out;
}

void Tensor::make_sub_node(TensorPtr a, TensorPtr b, TensorPtr out) {
  out->make_gradient();
  out->grad_node = std::make_shared<Node>(filterParents({a, b}), [a, b, out]() {
    std::vector<TensorPtr> p{out->grad};
    if (a->requires_grad) {
      p.push_back(a->grad);
    }
    if (b->requires_grad) {
      p.push_back(b->grad);
    }
    const DeviceTag dtag = get_dtag_by_presidence(p);
    TensorPtr out_grad = out->grad->cast(dtag);
    if (a->requires_grad) {
      TensorPtr a_grad = a->grad->cast(dtag);
      a_grad->upcast(out_grad->storage->dtype);
      a_grad->match_shape(out_grad);
      a_grad->materialize_broadcast();
      Weed::add_in_place(*(a_grad.get()), *(out_grad.get()));
      a->grad = a_grad;
      a->reduce_grad_broadcast();
    }
    if (b->requires_grad) {
      TensorPtr b_grad = b->grad->cast(dtag);
      b_grad->upcast(out_grad->storage->dtype);
      b_grad->match_shape(out_grad);
      b_grad->materialize_broadcast();
      Weed::sub_in_place(*(b_grad.get()), *(out_grad.get()));
      b->grad = b_grad;
      b->reduce_grad_broadcast();
    }
  });
}

TensorPtr Tensor::div(TensorPtr a, TensorPtr b) {
  const DeviceTag dtag = get_dtag_by_presidence({a, b});
  a->cast_in_place(dtag);
  b->cast_in_place(dtag);
  const bool rg = a->requires_grad || b->requires_grad;
  const bool s = IS_SPARSE(a) && IS_SPARSE(b);
  const DType dt = get_dtype_by_presidence({a, b});
  if (!a->match_shape(b) && !b->match_shape(a)) {
    throw std::invalid_argument("Tensor::match_shape() failed! (You tried to "
                                "alter an index that was not broadcast.)");
  }
  TensorPtr out = Tensor::allocate_like(a->shape, *(a.get()), dt, rg, s);

  Weed::div(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_div_node(a, b, out);
  }

  return out;
}

void Tensor::make_div_node(TensorPtr a, TensorPtr b, TensorPtr out) {
  out->make_gradient();
  out->grad_node = std::make_shared<Node>(filterParents({a, b}), [a, b, out]() {
    std::vector<TensorPtr> p{b, out->grad};
    if (a->requires_grad) {
      p.push_back(a->grad);
    }
    if (b->requires_grad) {
      p.push_back(b->grad);
      p.push_back(a);
    }
    const DeviceTag dtag = get_dtag_by_presidence(p);
    TensorPtr _b = b->cast(dtag);
    TensorPtr out_grad = out->grad->cast(dtag);
    if (a->requires_grad) {
      TensorPtr a_grad = a->grad->cast(dtag);
      const DType &dt = get_dtype_by_presidence({_b, out_grad});
      a_grad->upcast(dt);
      a_grad->match_shape(out_grad);
      a_grad->materialize_broadcast();
      TensorPtr tmp =
          Tensor::allocate_like(*(_b.get()), dt, false, IS_SPARSE(b));
      Weed::div(*(out_grad.get()), *(_b.get()), *(tmp.get()));
      Weed::add_in_place(*(a_grad.get()), *(tmp.get()));
      a->grad = a_grad;
      a->reduce_grad_broadcast();
    }
    if (b->requires_grad) {
      TensorPtr _a = a->cast(dtag);
      TensorPtr b_grad = b->grad->cast(dtag);
      TensorPtr b_sqr = Tensor::allocate_like(*(_b.get()), _b->storage->dtype,
                                              false, IS_SPARSE(b));
      Weed::mul(*(_b.get()), *(_b.get()), *(b_sqr.get()));
      const DType &dt = get_dtype_by_presidence({a, b_sqr});
      b_grad->upcast(dt);
      b_grad->match_shape(out_grad);
      b_grad->materialize_broadcast();
      TensorPtr tmp =
          Tensor::allocate_like(*(_a.get()), dt, false, IS_SPARSE(a));
      Weed::div(*(_a.get()), *(b_sqr.get()), *(tmp.get()));
      Weed::sub_in_place(*(b_grad.get()), *(tmp.get()));
      b->grad = b_grad;
      b->reduce_grad_broadcast();
    }
  });
}

TensorPtr Tensor::pow(TensorPtr a, real1 p) {
  const bool rg = a->requires_grad;
  TensorPtr out =
      allocate_like(*(a.get()), a->storage->dtype, rg, IS_SPARSE(a));

  Weed::pow(*(a.get()), p, *(out.get()));

  if (rg) {
    make_pow_node(a, p, out);
  }

  return out;
}

void Tensor::make_pow_node(TensorPtr x, real1 p, TensorPtr y) {
  y->make_gradient();
  y->grad_node = std::make_shared<Node>(std::vector<TensorPtr>{x}, [x, p, y]() {
    const DeviceTag dtag = get_dtag_by_presidence({x, y, x->grad, y->grad});

    TensorPtr dx = x->grad->cast(dtag);
    TensorPtr dy = y->grad->cast(dtag);

    dx->match_shape(dy);
    dx->materialize_broadcast();

    TensorPtr _x = x->cast(dtag);
    TensorPtr _y = y->cast(dtag);

    TensorPtr dy_y = Tensor::allocate_like(*(dy.get()), dy->storage->dtype,
                                           false, IS_SPARSE(dy));
    Weed::mul(*(dy.get()), *(_y.get()), *(dy_y.get()));

    TensorPtr s = SCALAR(p, dy_y);
    TensorPtr dy_y_p = s * dy_y;

    TensorPtr r = Tensor::allocate_like(*(dy_y_p.get()), dy_y_p->storage->dtype,
                                        false, IS_SPARSE(dy_y_p));
    Weed::div(*(dy_y_p.get()), *(_x.get()), *(r.get()));

    dx->upcast(r->storage->dtype);
    Weed::add_in_place(*(dx.get()), *(r.get()));
    x->grad = dx;
    x->reduce_grad_broadcast();
  });
}

TensorPtr Tensor::exp(TensorPtr a, real1 b) {
  const bool rg = a->requires_grad;
  TensorPtr out =
      allocate_like(*(a.get()), a->storage->dtype, rg, IS_SPARSE(a));

  Weed::exp(*(a.get()), b, *(out.get()));

  if (rg) {
    make_exp_node(a, (real1)std::log((real1_s)b), out);
  }

  return out;
}

void Tensor::make_exp_node(TensorPtr x, real1 log_b, TensorPtr y) {
  y->make_gradient();
  y->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{x}, [x, log_b, y]() {
        const DeviceTag dtag = get_dtag_by_presidence({y, x->grad, y->grad});

        TensorPtr dx = x->grad->cast(dtag);
        TensorPtr dy = y->grad->cast(dtag);

        dx->match_shape(dy);
        dx->materialize_broadcast();

        TensorPtr _y = y->cast(dtag);

        TensorPtr s = SCALAR(log_b, dy);
        TensorPtr dy_v = s * dy;

        TensorPtr r = Tensor::allocate_like(*(dy_v.get()), dy_v->storage->dtype,
                                            false, IS_SPARSE(dy_v));
        Weed::mul(*(dy_v.get()), *(_y.get()), *(r.get()));

        dx->upcast(r->storage->dtype);
        Weed::add_in_place(*(dx.get()), *(r.get()));
        x->grad = dx;
        x->reduce_grad_broadcast();
      });
}

TensorPtr Tensor::log(TensorPtr a, real1 b) {
  const bool rg = a->requires_grad;
  TensorPtr out =
      allocate_like(*(a.get()), a->storage->dtype, rg, IS_SPARSE(a));

  Weed::log(*(a.get()), b, *(out.get()));

  if (rg) {
    make_log_node(a, (real1)(ONE_R1 / std::log((real1_s)b)), out);
  }

  return out;
}

void Tensor::make_log_node(TensorPtr x, real1 inv_log_b, TensorPtr y) {
  y->make_gradient();
  y->grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{x}, [x, inv_log_b, y]() {
        const DeviceTag dtag = get_dtag_by_presidence({x, x->grad, y->grad});

        TensorPtr dx = x->grad->cast(dtag);
        TensorPtr dy = y->grad->cast(dtag);

        dx->match_shape(dy);
        dx->materialize_broadcast();

        TensorPtr _x = x->cast(dtag);

        TensorPtr s = SCALAR(inv_log_b, dy);
        TensorPtr dy_v = s * dy;

        TensorPtr r = Tensor::allocate_like(*(dy_v.get()), dy_v->storage->dtype,
                                            false, IS_SPARSE(dy_v));
        Weed::div(*(dy_v.get()), *(_x.get()), *(r.get()));

        dx->upcast(r->storage->dtype);
        Weed::add_in_place(*(dx.get()), *(r.get()));
        x->grad = dx;
        x->reduce_grad_broadcast();
      });
}
} // namespace Weed
