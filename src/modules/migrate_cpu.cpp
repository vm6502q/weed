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

#include "modules/migrate_cpu.hpp"
#include "autograd/node.hpp"
#include "ops/in_place.hpp"

namespace Weed {
TensorPtr MigrateCpu::forward(const TensorPtr x) {
  if (!(x->storage->is_gpu())) {
    return x;
  }

  TensorPtr out = std::make_shared<Tensor>(*(x.get()));
  out->storage = out->storage->cpu();
  if (x->requires_grad) {
    out->make_gradient();
    out->grad_node =
        std::make_shared<Node>(std::vector<TensorPtr>{x}, [x, out] {
          const DeviceTag dtag =
              Tensor::get_dtag_by_presidence({x->grad, out->grad});
          TensorPtr x_grad = x->grad->cast(dtag);
          TensorPtr out_grad = out->grad->cast(dtag);
          Weed::add_in_place(*(x_grad.get()), *(out_grad.get()));
          x->grad = x_grad;
        });
  }

  return out;
}
ParameterPtr MigrateCpu::pforward(const ParameterPtr x) {
  if (!(x->storage->is_gpu())) {
    return x;
  }

  ParameterPtr out = std::make_shared<Parameter>(*(x.get()));
  out->storage = out->storage->cpu();
  if (x->requires_grad) {
    out->requires_grad = true;
    out->make_gradient();
    out->grad_node =
        std::make_shared<Node>(std::vector<TensorPtr>{x}, [x, out] {
          const DeviceTag dtag =
              Tensor::get_dtag_by_presidence({x->grad, out->grad});
          TensorPtr x_grad = x->grad->cast(dtag);
          TensorPtr out_grad = out->grad->cast(dtag);
          Weed::add_in_place(*(x_grad.get()), *(out_grad.get()));
          x->grad = x_grad;
        });
  } else {
    out->requires_grad = false;
  }

  return out;
}
} // namespace Weed
