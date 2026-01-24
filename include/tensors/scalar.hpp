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

#include "tensor.hpp"

namespace Weed {
struct Scalar;

typedef std::shared_ptr<Scalar> ScalarPtr;

/**
 * Tensor with only 1 element (with broadcast on tensor operations)
 */
struct Scalar : public Tensor {
  Scalar(real1 v, bool rg, DeviceTag dtag, int64_t did = -1)
      : Tensor(std::vector<real1>{v}, std::vector<vecCapInt>{ONE_VCI},
               std::vector<vecCapInt>{ZERO_VCI}, rg, dtag, did) {}
  Scalar(complex v, bool rg, DeviceTag dtag, int64_t did = -1)
      : Tensor(std::vector<complex>{v}, std::vector<vecCapInt>{ONE_VCI},
               std::vector<vecCapInt>{ZERO_VCI}, rg, dtag, did) {}
  Scalar(TensorPtr orig) {
    if (orig->get_size() > ONE_VCI) {
      throw std::invalid_argument(
          "Cannot construct scalar from Tensor with get_size() > 1!");
    }
    shape = std::vector<vecCapInt>{ONE_VCI};
    stride = std::vector<vecCapInt>{ZERO_VCI};
    offset = orig->offset;
    storage = orig->storage;
    grad_node = orig->grad_node;
    grad = orig->grad;
  }
};
} // namespace Weed
