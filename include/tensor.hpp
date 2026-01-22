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
#include "device_tag.hpp"
#include "dtype.hpp"

#include <vector>

namespace Weed {
struct Tensor;

typedef std::shared_ptr<Tensor> TensorPtr;

struct Tensor : public std::enable_shared_from_this<Tensor> {
  StoragePtr storage;

  std::vector<vecCapIntGpu> shape;
  std::vector<vecCapIntGpu> stride;
  vecCapIntGpu offset;

  bool requires_grad;
  NodePtr grad_node;
  TensorPtr grad;

  Tensor()
      : storage(nullptr), shape(), stride(), offset(0U), requires_grad(false),
        grad_node(nullptr), grad(nullptr) {}
  Tensor(std::vector<vecCapIntGpu> shp, std::vector<vecCapIntGpu> strd,
         bool rg = false, DType dtype = DType::REAL,
         DeviceTag dtag = DeviceTag::CPU, int64_t did = -1);

  TensorPtr get_ptr() { return shared_from_this(); }

  vecCapIntGpu get_size() const {
    if (shape.empty()) {
      return 0U;
    }
    vecCapIntGpu max_index = offset;
    for (size_t i = 0; i < shape.size(); ++i) {
      max_index += (shape[i] - 1U) * stride[i];
    }

    return max_index + 1U;
  }

  // Shallow copy:
  Tensor copy() {
    Tensor cp;
    // A tensor is a view on storage:
    cp.storage = storage;
    cp.shape = shape;
    cp.stride = stride;
    cp.offset = offset;
    cp.requires_grad = requires_grad;
    cp.grad_node = grad_node;
    cp.grad = grad;

    return cp;
  }

  static Tensor allocate_like(const Tensor &orig, const DType &dt,
                              const bool &rg);
  static Tensor allocate_like(const std::vector<vecCapIntGpu> &shape,
                              const std::vector<vecCapIntGpu> &stride,
                              const Tensor &orig, const DType &dt,
                              const bool &rg);

  static Tensor transpose(Tensor &a);

  static Tensor relu(Tensor &a);
  static Tensor add(Tensor &a, Tensor &b);
  static Tensor mul(Tensor &a, Tensor &b);
  static Tensor matmul(Tensor &a, Tensor &b);
};
} // namespace Weed
