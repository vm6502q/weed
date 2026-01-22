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

Tensor Tensor::allocate_like(const Tensor &orig, const DType &dt) {
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
  return Tensor(orig.shape, orig.stride, false, dt, dtag, did);
}

Tensor Tensor::allocate_like(const std::vector<vecCapIntGpu> &shape,
                             const std::vector<vecCapIntGpu> &stride,
                             const Tensor &orig, const DType &dt) {
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
  return Tensor(shape, stride, false, dt, dtag, did);
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
    grad->storage->FillZero();
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
  Tensor out = allocate_like(a, a.storage->dtype);

  Weed::relu(a, out);

  if (!a.requires_grad) {
    return out;
  }

  out.requires_grad = true;

  out.grad_node = std::make_shared<Node>(
      std::vector<TensorPtr>{a.get_ptr()},
      [out](std::vector<TensorPtr> parents) {
        for (TensorPtr in : parents) {
          relu_grad(*(in->grad.get()), *(in.get()), *(out.grad.get()));
        }
      });

  return out;
}

Tensor Tensor::add(Tensor &a, Tensor &b) {
  Tensor out = allocate_like(a, get_dtype_by_presidence(a, b));

  Weed::add(a, b, out);

  if (!a.requires_grad && !b.requires_grad) {
    return out;
  }

  out.requires_grad = true;

  out.grad_node = std::make_shared<Node>(
      filterParents({a.get_ptr(), b.get_ptr()}),
      [out](std::vector<TensorPtr> parents) {
        for (TensorPtr in : parents) {
          add_inplace(*(in->grad.get()), *(out.grad.get()));
        }
      });

  return out;
}

Tensor Tensor::mul(Tensor &a, Tensor &b) {
  Tensor out = allocate_like(a, get_dtype_by_presidence(a, b));

  Weed::mul(a, b, out);

  if (!a.requires_grad && !b.requires_grad) {
    return out;
  }

  out.requires_grad = true;

  out.grad_node = std::make_shared<Node>(
      std::vector<TensorPtr>{a.get_ptr(), b.get_ptr()},
      [out](std::vector<TensorPtr> parents) {
        Tensor &a = *(parents[0U].get());
        Tensor &b = *(parents[1U].get());
        if (a.requires_grad) {
          add_inplace(*(a.grad.get()), Tensor::mul(*(out.grad.get()), b));
        }
        if (b.requires_grad) {
          add_inplace(*(b.grad.get()), Tensor::mul(*(out.grad.get()), a));
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
  Tensor out = allocate_like(shp, str, a, get_dtype_by_presidence(a, b));

  Weed::matmul(a, b, out);

  if (!a.requires_grad && !b.requires_grad) {
    return out;
  }

  out.requires_grad = true;

  out.grad_node = std::make_shared<Node>(
      std::vector<TensorPtr>{a.get_ptr(), b.get_ptr()},
      [out](std::vector<TensorPtr> parents) {
        Tensor &a = *(parents[0U].get());
        Tensor &b = *(parents[1U].get());
        if (a.requires_grad) {
          Tensor bt = transpose(b);
          add_inplace(*(a.grad.get()), matmul(*(out.grad.get()), bt));
        }
        if (b.requires_grad) {
          Tensor at = transpose(a);
          add_inplace(*(b.grad.get()), matmul(at, *(out.grad.get())));
        }
      });

  return out;
}
} // namespace Weed
