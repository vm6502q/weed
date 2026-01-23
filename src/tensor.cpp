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
Tensor Tensor::allocate_like(const Tensor &orig, const DType &dt,
                             const bool &rg) {
  const StoragePtr storage_ptr = orig.storage;
  const DeviceTag dtag = storage_ptr->device;
  const int64_t did = storage_ptr->get_device_id();

  return Tensor(orig.shape, orig.stride, rg, dt, dtag, did);
}

Tensor Tensor::allocate_like(const std::vector<vecCapInt> &shape,
                             const std::vector<vecCapInt> &stride,
                             const Tensor &orig, const DType &dt,
                             const bool &rg) {
  const StoragePtr storage_ptr = orig.storage;
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
  return Tensor(shape, stride, rg, dt, dtag, did);
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

Tensor Tensor::operator[](vecCapInt idx) {
  if (idx >= shape.back()) {
    throw std::invalid_argument("Tensor index out-of-range!");
  }

  Tensor v = copy();
  v.offset += idx * v.stride.back();
  v.shape.pop_back();
  v.stride.pop_back();

  if (v.grad) {
    v.grad = (*(v.grad))[idx].get_ptr();
  }

  if (v.shape.empty()) {
    switch (v.storage->dtype) {
    case DType::COMPLEX:
      return ComplexScalar(v);
    case DType::REAL:
      return RealScalar(v);
    default:
      return Scalar(v);
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

void Tensor::backward(Tensor &loss) {
  if (!loss.requires_grad()) {
    return;
  }

  // Seed gradient of loss (scalar assumed)
  // loss.grad already allocated by our invariant
  loss.grad->storage->FillOnes();

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

  loss.grad_node->backward(loss.grad_node->parents);

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    (*it)->backward((*it)->parents);
  }
}

Tensor Tensor::transpose(Tensor &a) {
  if (a.shape.size() != 2U) {
    throw std::invalid_argument(
        "Tensor::tranpose is (currently) only for matrices with 2 indices!");
  }

  // Shallow copy (keeps storage and gradient)
  Tensor out = a.copy();
  // Change tensor view:
  std::swap(out.shape[0U], out.shape[1U]);
  std::swap(out.stride[0U], out.stride[1U]);

  return out;
}

Tensor Tensor::abs(Tensor &a) {
  const bool rg = a.requires_grad();
  Tensor out = allocate_like(a, a.storage->dtype, rg);

  Weed::abs(a, out);

  if (rg) {
    make_abs_node(a, out);
  }

  return out;
}

void Tensor::make_abs_node(Tensor &a, Tensor &out) {
  out.grad_node =
      std::make_shared<Node>(std::vector<TensorPtr>{a.get_ptr()},
                             [out](std::vector<TensorPtr> parents) {
                               Tensor &out_grad = *(out.grad.get());
                               const DType &dt = out_grad.storage->dtype;
                               for (TensorPtr in : parents) {
                                 Tensor &in_grad = *(in->grad.get());
                                 in_grad.upcast(dt);
                                 Weed::abs_grad(in_grad, *(in.get()), out_grad);
                               }
                             });
}

Tensor Tensor::relu(Tensor &a) {
  const bool rg = a.requires_grad();
  Tensor out = allocate_like(a, a.storage->dtype, rg);

  Weed::relu(a, out);

  if (rg) {
    make_relu_node(a, out);
  }

  return out;
}

void Tensor::make_relu_node(Tensor &a, Tensor &out) {
  out.grad_node = std::make_shared<Node>(
      std::vector<TensorPtr>{a.get_ptr()},
      [out](std::vector<TensorPtr> parents) {
        Tensor &out_grad = *(out.grad.get());
        const DType &dt = out_grad.storage->dtype;
        for (TensorPtr in : parents) {
          Tensor &in_grad = *(in->grad.get());
          in_grad.upcast(dt);
          Weed::relu_grad(in_grad, *(in.get()), out_grad);
        }
      });
}

Tensor Tensor::add(Tensor &a, Tensor &b) {
  const bool rg = a.requires_grad() || b.requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  Tensor out = allocate_like(a, dt, rg);

  Weed::add(a, b, out);

  if (rg) {
    make_add_node(a, b, out);
  }

  return out;
}

void Tensor::make_add_node(Tensor &a, Tensor &b, Tensor &out) {
  out.grad_node =
      std::make_shared<Node>(filterParents({a.get_ptr(), b.get_ptr()}),
                             [out](std::vector<TensorPtr> parents) {
                               Tensor &out_grad = *(out.grad.get());
                               const DType &dt = out_grad.storage->dtype;
                               for (TensorPtr in : parents) {
                                 Tensor &in_grad = *(in->grad.get());
                                 Tensor n_out =
                                     Tensor::allocate_like(in_grad, dt, false);
                                 Weed::add(in_grad, out_grad, n_out);
                                 in->grad = n_out.get_ptr();
                               }
                             });
}

Tensor Tensor::mul(Tensor &a, Tensor &b) {
  const bool rg = a.requires_grad() || b.requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  Tensor out = allocate_like(a, dt, rg);

  Weed::mul(a, b, out);

  if (rg) {
    make_mul_node(a, b, out);
  }

  return out;
}

void Tensor::make_mul_node(Tensor &a, Tensor &b, Tensor &out) {
  out.grad_node = std::make_shared<Node>(
      std::vector<TensorPtr>{a.get_ptr(), b.get_ptr()},
      [out](std::vector<TensorPtr> parents) {
        Tensor &out_grad = *(out.grad.get());
        const DType &dt = out_grad.storage->dtype;
        Tensor &a = *(parents[0U].get());
        Tensor &b = *(parents[1U].get());
        if (a.requires_grad()) {
          Tensor tmp = Tensor::allocate_like(b, dt, false);
          Tensor &a_grad = *(a.grad.get());
          a_grad.upcast(dt);
          Weed::mul(out_grad, b, tmp);
          Tensor n_out = Tensor::allocate_like(tmp, dt, false);
          Weed::add(a_grad, tmp, n_out);
          a.grad = n_out.get_ptr();
        }
        if (b.requires_grad()) {
          Tensor tmp = Tensor::allocate_like(a, dt, false);
          Tensor &b_grad = *(b.grad.get());
          b_grad.upcast(dt);
          Weed::mul(out_grad, a, tmp);
          Tensor n_out = Tensor::allocate_like(tmp, dt, false);
          Weed::add(b_grad, tmp, n_out);
          b.grad = n_out.get_ptr();
        }
      });
}

Tensor Tensor::matmul(Tensor &a, Tensor &b) {
  if ((a.shape.size() != 2U) || (b.shape.size() != 2U)) {
    throw std::invalid_argument(
        "Tensor::matmul is only for matrices with 2 indices!");
  }

  const std::vector<vecCapInt> shp = {a.shape[0U], b.shape[1U]};
  const std::vector<vecCapInt> str = {ONE_VCI, a.shape[0U]};
  const bool rg = a.requires_grad() || b.requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  Tensor out = allocate_like(shp, str, a, dt, rg);

  Weed::matmul(a, b, out);

  if (rg) {
    make_matmul_node(a, b, out);
  }

  return out;
}

void Tensor::make_matmul_node(Tensor &a, Tensor &b, Tensor &out) {
  out.grad_node = std::make_shared<Node>(
      std::vector<TensorPtr>{a.get_ptr(), b.get_ptr()},
      [out](std::vector<TensorPtr> parents) {
        Tensor &out_grad = *(out.grad.get());
        const DType &dt = out_grad.storage->dtype;
        Tensor &a = *(parents[0U].get());
        Tensor &b = *(parents[1U].get());
        if (a.requires_grad()) {
          Tensor bt = transpose(b);
          Tensor tmp = Tensor::allocate_like(a, dt, false);
          Tensor &a_grad = *(a.grad.get());
          a_grad.upcast(dt);
          Weed::matmul(*(out.grad.get()), bt, tmp);
          Tensor n_out = Tensor::allocate_like(tmp, dt, false);
          Weed::add(a_grad, tmp, n_out);
          a.grad = n_out.get_ptr();
        }
        if (b.requires_grad()) {
          Tensor at = transpose(a);
          Tensor tmp = Tensor::allocate_like(b, dt, false);
          Tensor &b_grad = *(b.grad.get());
          b_grad.upcast(dt);
          Weed::matmul(at, *(out.grad.get()), tmp);
          Tensor n_out = Tensor::allocate_like(tmp, dt, false);
          Weed::add(b_grad, tmp, n_out);
          b.grad = n_out.get_ptr();
        }
      });
}
} // namespace Weed
