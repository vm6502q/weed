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
#include "add.hpp"
#include "mul.hpp"
#include "cpu_complex_storage.hpp"
#include "cpu_real_storage.hpp"
#include "node.hpp"

namespace Weed {
void Tensor::validate() const {
  if (!storage) {
    throw std::logic_error("Tensor has null storage");
  }

  if (shape.size() != stride.size()) {
    throw std::logic_error("Tensor shape/stride rank mismatch");
  }

  // offset must be in-bounds
  if (offset >= storage->size) {
    throw std::out_of_range("Tensor offset exceeds storage size");
  }

  // Compute maximum reachable index
  vecCapIntGpu max_index = offset;

  for (size_t d = 0; d < shape.size(); ++d) {
    if (shape[d] == 0) {
      // Empty tensor: safe by definition
      return;
    }

    // Guard against overflow if you want to be extra careful
    vecCapIntGpu extent = shape[d] - 1;
    max_index += extent * stride[d];
  }

  if (max_index >= storage->size) {
    throw std::out_of_range("Tensor view exceeds storage bounds");
  }
}

Tensor Tensor::allocate_like(const Tensor &orig) {
  Tensor out;
  out.requires_grad = orig.requires_grad;
  vecCapIntGpu size = 0U;
  for (size_t i = 0U; i < shape.size(); ++i) {
    size += (orig.shape[i] - 1U) * orig.stride[i];
  }
  switch (orig.storage->dtype) {
  case DType::COMPLEX:
    out.storage = std::make_shared<CpuComplexStorage>(size);
  case DType::REAL:
  default:
    out.storage = std::make_shared<CpuRealStorage>(size);
  }

  return out;
}

std::vector<const Tensor *> filterParents(std::vector<const Tensor *> parents) {
  std::vector<const Tensor *> filtered;
  for (const Tensor *p : parents) {
    if (p->requires_grad) {
      filtered.push_back(p);
    }
  }

  return filtered;
}

Tensor Tensor::add(const Tensor &a, const Tensor &b) {
  Tensor out = allocate_like(a);

  Weed::add(a, b, out);

  if (!a.requires_grad && !b.requires_grad) {
    return out;
  }

  const std::vector<const Tensor *> parents = filterParents({ &a, &b });
  out.grad_node = std::make_shared<Node>(parents, [=]() {
    for (const Tensor *in : parents) {
      add_inplace(in->grad, out.grad);
    }
  });

  return out;
}

Tensor Tensor::mul(const Tensor &a, const Tensor &b) {
  Tensor out = allocate_like(a);

  Weed::mul(a, b, out);

  if (!a.requires_grad && !b.requires_grad) {
    return out;
  }

  const std::vector<const Tensor *> parents = filterParents({ &a, &b });
  out.grad_node = std::make_shared<Node>(parents, [=]() {
    for (const Tensor *in : parents) {
      mul_inplace(in->grad, out.grad);
    }
  });

  return out;
}
} // namespace Weed
