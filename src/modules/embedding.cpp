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

#include "modules/embedding.hpp"
#include "autograd/node.hpp"
#include "ops/embedding.hpp"

namespace Weed {
TensorPtr Embedding::forward(const SymbolTensorPtr indices) {
  // Output shape = indices.shape + [embedding_dim]
  std::vector<tcapint> out_shape = indices->shape;
  out_shape.push_back(embedding_dim);
  std::vector<tcapint> out_stride = Tensor::full_contiguous_stride(out_shape);

  TensorPtr out = Tensor::allocate_like(
      out_shape, out_stride, *(weight.get()), weight->storage->dtype,
      weight->requires_grad, weight->storage->is_sparse());

  Weed::embedding_gather(*(indices.get()), *(weight.get()), *(out.get()));

  ParameterPtr w = weight;

  out->make_gradient();
  out->grad_node = std::make_shared<
      Node>(std::vector<TensorPtr>{weight}, [indices, w, out]() {
    const DeviceTag dtag = Tensor::get_dtag_by_presidence({w->grad, out->grad});
    TensorPtr dW = w->grad->cast(dtag);
    TensorPtr dout = out->grad->cast(dtag);
    Weed::embedding_scatter_add(*(dW.get()), *(indices.get()), *(dout.get()));
    w->grad = dW;
  });

  return out;
}
} // namespace Weed
