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
#include "common/serializer.hpp"
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
  out->grad_node = std::make_shared<Node>(
      std::vector<TensorPtr>{weight}, [indices, w, out]() {
        const DeviceTag dtag =
            Tensor::get_dtag_by_presidence({w->grad, out->grad, indices});
        TensorPtr dW = w->grad->cast(dtag);
        TensorPtr dout =
            std::make_shared<Tensor>(*(out->grad.get()))->cast(dtag);
        SymbolTensorPtr _indices = indices->cast(dtag);

        dout->match_shape(w);
        dW->match_shape(w);
        dW->materialize_broadcast();

        dW->upcast(dout->storage->dtype);
        Weed::embedding_scatter_add(*(dW.get()), *(_indices.get()),
                                    *(dout.get()));
        w->grad = dW;
        w->reduce_grad_broadcast();
      });

  return out;
}
void Embedding::save(std::ostream &os) const {
  Module::save(os);
  Serializer::write_tcapint(os, num_embeddings);
  Serializer::write_tcapint(os, embedding_dim);
  weight->save(os);
}
} // namespace Weed
