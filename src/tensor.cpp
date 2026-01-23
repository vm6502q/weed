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

#include "complex_scalar.hpp"
#include "node.hpp"
#include "real_scalar.hpp"

#include "abs.hpp"
#include "add.hpp"
#include "matmul.hpp"
#include "mul.hpp"
#include "relu.hpp"

#include "cpu_complex_storage.hpp"
#include "cpu_real_storage.hpp"
#include "gpu_complex_storage.hpp"
#include "gpu_real_storage.hpp"

#include <unordered_set>

#define PICK_DEVICE_STORAGE(GpuType, CpuType)                                  \
  switch (dtag) {                                                              \
  case DeviceTag::GPU:                                                         \
    storage = std::make_shared<GpuType>(size, did);                            \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    storage = std::make_shared<CpuType>(size);                                 \
  }                                                                            \
  break

namespace Weed {
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
    PICK_DEVICE_STORAGE(GpuComplexStorage, CpuComplexStorage);
    break;
  case DType::REAL:
  default:
    PICK_DEVICE_STORAGE(GpuRealStorage, CpuRealStorage);
  }

  if (rg) {
    grad->storage->FillZeros();
  }
}

TensorPtr Tensor::operator[](vecCapInt idx) {
  if (idx >= shape.back()) {
    throw std::invalid_argument("Tensor index out-of-range!");
  }

  TensorPtr v = copy();
  v->offset += idx * v->stride.back();
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
        p->grad_node->backward(p->grad_node->parents);
      }
    }
    topo.push_back(n);
  };

  loss->grad_node->backward(loss->grad_node->parents);

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->backward((*it)->parents);
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

TensorPtr Tensor::abs(TensorPtr a) {
  const bool rg = a->requires_grad();
  TensorPtr out = allocate_like(a, a->storage->dtype, rg);

  Weed::abs(*(a.get()), *(out.get()));

  if (rg) {
    make_abs_node(a, out);
  }

  return out;
}

void Tensor::make_abs_node(TensorPtr a, TensorPtr out) {
  out->grad_node = std::make_shared<Node>(
      std::vector<TensorPtr>{a}, [out](std::vector<TensorPtr> parents) {
        Tensor &out_grad = *(out->grad.get());
        const DType &dt = out_grad.storage->dtype;
        for (TensorPtr in : parents) {
          Tensor &in_grad = *(in->grad.get());
          in_grad.upcast(dt);
          Weed::abs_grad(in_grad, *(in.get()), out_grad);
        }
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
  out->grad_node = std::make_shared<Node>(
      std::vector<TensorPtr>{a}, [out](std::vector<TensorPtr> parents) {
        Tensor &out_grad = *(out->grad.get());
        const DType &dt = out_grad.storage->dtype;
        for (TensorPtr in : parents) {
          Tensor &in_grad = *(in->grad.get());
          in_grad.upcast(dt);
          Weed::relu_grad(in_grad, *(in.get()), out_grad);
        }
      });
}

TensorPtr Tensor::add(TensorPtr a, TensorPtr b) {
  const bool rg = a->requires_grad() || b->requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  TensorPtr out = allocate_like(a, dt, rg);

  Weed::add(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_add_node(a, b, out);
  }

  return out;
}

void Tensor::make_add_node(TensorPtr a, TensorPtr b, TensorPtr out) {
  out->grad_node = std::make_shared<Node>(
      filterParents({
          a,
      }),
      [out](std::vector<TensorPtr> parents) {
        TensorPtr out_grad = out->grad;
        const DType &dt = out_grad->storage->dtype;
        for (TensorPtr in : parents) {
          TensorPtr in_grad = in->grad;
          TensorPtr n_out = Tensor::allocate_like(in_grad, dt, false);
          Weed::add(*(in_grad.get()), *(out_grad.get()), *(n_out.get()));
          in->grad = n_out;
        }
      });
}

TensorPtr Tensor::mul(TensorPtr a, TensorPtr b) {
  const bool rg = a->requires_grad() || b->requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  TensorPtr out = allocate_like(a, dt, rg);

  Weed::mul(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_mul_node(a, b, out);
  }

  return out;
}

void Tensor::make_mul_node(TensorPtr a, TensorPtr b, TensorPtr out) {
  out->grad_node = std::make_shared<Node>(
      std::vector<TensorPtr>{a, b}, [out](std::vector<TensorPtr> parents) {
        TensorPtr out_grad = out->grad;
        const DType &dt = out_grad->storage->dtype;
        TensorPtr a = parents[0U];
        TensorPtr b = parents[1U];
        if (a->requires_grad()) {
          TensorPtr tmp = Tensor::allocate_like(b, dt, false);
          TensorPtr a_grad = a->grad;
          a_grad->upcast(dt);
          Weed::mul(*(out_grad.get()), *(b.get()), *(tmp.get()));
          TensorPtr n_out = Tensor::allocate_like(tmp, dt, false);
          Weed::add(*(a_grad.get()), *(tmp.get()), *(n_out.get()));
          a->grad = n_out;
        }
        if (b->requires_grad()) {
          TensorPtr tmp = Tensor::allocate_like(a, dt, false);
          TensorPtr b_grad = b->grad;
          b_grad->upcast(dt);
          Weed::mul(*(out_grad.get()), *(a.get()), *(tmp.get()));
          TensorPtr n_out = Tensor::allocate_like(tmp, dt, false);
          Weed::add(*(b_grad.get()), *(tmp.get()), *(n_out.get()));
          b->grad = n_out;
        }
      });
}

TensorPtr Tensor::matmul(TensorPtr a, TensorPtr b) {
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
      std::vector<TensorPtr>{a, b}, [out](std::vector<TensorPtr> parents) {
        TensorPtr out_grad = out->grad;
        const DType &dt = out_grad->storage->dtype;
        TensorPtr a = parents[0U];
        TensorPtr b = parents[1U];
        if (a->requires_grad()) {
          TensorPtr bt = transpose(b);
          TensorPtr tmp = Tensor::allocate_like(a, dt, false);
          TensorPtr a_grad = a->grad;
          a_grad->upcast(dt);
          Weed::matmul(*(out->grad.get()), *(bt.get()), *(tmp.get()));
          TensorPtr n_out = Tensor::allocate_like(tmp, dt, false);
          Weed::add(*(a_grad.get()), *(tmp.get()), *(n_out.get()));
          a->grad = n_out;
        }
        if (b->requires_grad()) {
          TensorPtr at = transpose(a);
          TensorPtr tmp = Tensor::allocate_like(b, dt, false);
          TensorPtr b_grad = b->grad;
          b_grad->upcast(dt);
          Weed::matmul(*(at.get()), *(out->grad.get()), *(tmp.get()));
          TensorPtr n_out = Tensor::allocate_like(tmp, dt, false);
          Weed::add(*(b_grad.get()), *(tmp.get()), *(n_out.get()));
          b->grad = n_out;
        }
      });
}
} // namespace Weed
