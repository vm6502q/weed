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

#include "tensor.hpp"
#include "node.hpp"

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
inline DType get_dtype_by_presidence(const Tensor &left, const Tensor &right) {
  if (right.storage->dtype == DType::COMPLEX) {
    return DType::COMPLEX;
  }
  return left.storage->dtype;
}

Tensor Tensor::allocate_like(const Tensor &orig, const DType &dt,
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
  return Tensor(orig.shape, orig.stride, rg, dt, dtag, did);
}

Tensor Tensor::allocate_like(const std::vector<vecCapIntGpu> &shape,
                             const std::vector<vecCapIntGpu> &stride,
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

Tensor::Tensor(std::vector<vecCapIntGpu> shp, std::vector<vecCapIntGpu> strd,
               bool rg, DType dtype, DeviceTag dtag, int64_t did)
    : shape(shp), stride(strd), offset(0U), requires_grad(rg),
      grad_node(nullptr), grad(nullptr) {
  if (shape.size() != stride.size()) {
    throw std::invalid_argument(
        "Tensor shape vector must have same length as stride vector!");
  }

  const vecCapIntGpu size = get_size();

  switch (dtype) {
  case DType::COMPLEX:
    PICK_DEVICE_STORAGE(GpuComplexStorage, CpuComplexStorage);
    break;
  case DType::REAL:
  default:
    PICK_DEVICE_STORAGE(GpuRealStorage, CpuRealStorage);
  }

  if (requires_grad) {
    grad = std::make_shared<Tensor>(shape, stride, false, dtype, dtag, did);
    grad->storage->FillZeros();
  }
}

std::vector<TensorPtr> filterParents(std::vector<TensorPtr> parents) {
  std::vector<TensorPtr> filtered;
  for (TensorPtr p : parents) {
    if (p->requires_grad) {
      filtered.push_back(p);
    }
  }

  return filtered;
}

void Tensor::backward(Tensor &loss) {
  if (!loss.requires_grad) {
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

Tensor Tensor::relu(Tensor &a) {
  const bool rg = a.requires_grad;
  Tensor out = allocate_like(a, a.storage->dtype, rg);

  Weed::relu(a, out);

  if (!rg) {
    return out;
  }

  out.grad_node = std::make_shared<Node>(
      std::vector<TensorPtr>{a.get_ptr()},
      [out](std::vector<TensorPtr> parents) {
        for (TensorPtr in : parents) {
          Weed::relu_grad(*(in->grad.get()), *(in.get()), *(out.grad.get()));
        }
      });

  return out;
}

Tensor Tensor::add(Tensor &a, Tensor &b) {
  const bool rg = a.requires_grad || b.requires_grad;
  DType dt = get_dtype_by_presidence(a, b);
  Tensor out = allocate_like(a, dt, rg);

  Weed::add(a, b, out);

  if (!rg) {
    return out;
  }

  out.grad_node =
      std::make_shared<Node>(filterParents({a.get_ptr(), b.get_ptr()}),
                             [dt, out](std::vector<TensorPtr> parents) {
                               Tensor &out_grad = *(out.grad.get());
                               for (TensorPtr in : parents) {
                                 Tensor &in_grad = *(in->grad.get());
                                 in_grad.upcast(dt);
                                 Weed::add_inplace(in_grad, out_grad);
                               }
                             });

  return out;
}

Tensor Tensor::mul(Tensor &a, Tensor &b) {
  const bool rg = a.requires_grad || b.requires_grad;
  DType dt = get_dtype_by_presidence(a, b);
  Tensor out = allocate_like(a, dt, rg);

  Weed::mul(a, b, out);

  if (!rg) {
    return out;
  }

  out.grad_node = std::make_shared<Node>(
      std::vector<TensorPtr>{a.get_ptr(), b.get_ptr()},
      [dt, out](std::vector<TensorPtr> parents) {
        Tensor &a = *(parents[0U].get());
        Tensor &b = *(parents[1U].get());
        if (a.requires_grad) {
          Tensor tmp = Tensor::allocate_like(b, dt, false);
          Tensor &a_grad = *(a.grad.get());
          a_grad.upcast(dt);
          Weed::mul(*(out.grad.get()), b, tmp);
          Weed::add_inplace(a_grad, tmp);
        }
        if (b.requires_grad) {
          Tensor tmp = Tensor::allocate_like(a, dt, false);
          Tensor &b_grad = *(b.grad.get());
          b_grad.upcast(dt);
          Weed::mul(*(out.grad.get()), a, tmp);
          Weed::add_inplace(b_grad, tmp);
        }
      });

  return out;
}

Tensor Tensor::matmul(Tensor &a, Tensor &b) {
  if ((a.shape.size() != 2U) || (b.shape.size() != 2U)) {
    throw std::invalid_argument(
        "Tensor::matmul is only for matrices with 2 indices!");
  }

  const std::vector<vecCapIntGpu> shp = {a.shape[0U], b.shape[1U]};
  const std::vector<vecCapIntGpu> str = {1U, a.shape[0U]};
  const bool rg = a.requires_grad || b.requires_grad;
  DType dt = get_dtype_by_presidence(a, b);
  Tensor out = allocate_like(shp, str, a, dt, rg);

  Weed::matmul(a, b, out);

  if (!rg) {
    return out;
  }

  out.grad_node = std::make_shared<Node>(
      std::vector<TensorPtr>{a.get_ptr(), b.get_ptr()},
      [dt, out](std::vector<TensorPtr> parents) {
        Tensor &a = *(parents[0U].get());
        Tensor &b = *(parents[1U].get());
        if (a.requires_grad) {
          Tensor bt = transpose(b);
          Tensor tmp = Tensor::allocate_like(a, dt, false);
          Tensor &a_grad = *(a.grad.get());
          a_grad.upcast(dt);
          Weed::matmul(*(out.grad.get()), bt, tmp);
          Weed::add_inplace(a_grad, tmp);
        }
        if (b.requires_grad) {
          Tensor at = transpose(a);
          Tensor tmp = Tensor::allocate_like(b, dt, false);
          Tensor &b_grad = *(b.grad.get());
          b_grad.upcast(dt);
          Weed::matmul(at, *(out.grad.get()), tmp);
          Weed::add_inplace(b_grad, tmp);
        }
      });

  return out;
}
} // namespace Weed
