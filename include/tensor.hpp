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
  StoragePtr grad;

  std::vector<vecCapIntGpu> shape;
  std::vector<vecCapIntGpu> stride;
  vecCapIntGpu offset;

  bool requires_grad;
  NodePtr grad_node;

  Tensor(std::vector<vecCapIntGpu> shp, std::vector<vecCapIntGpu> strd,
         bool rg = false, DType dtype = DType::REAL,
         DeviceTag dtag = DeviceTag::CPU, int64_t did = -1);

  TensorPtr get_ptr() { return shared_from_this(); }

  vecCapIntGpu get_size() const {
    vecCapIntGpu size = 0U;
    for (size_t i = 0U; i < shape.size(); ++i) {
      size += (shape[i] - 1U) * stride[i];
    }

    return size;
  }

  static Tensor allocate_like(const Tensor &orig, const DType &dt);

  static Tensor relu(Tensor &a);
  static Tensor add(Tensor &a, Tensor &b);
  static Tensor mul(Tensor &a, Tensor &b);
};
} // namespace Weed
