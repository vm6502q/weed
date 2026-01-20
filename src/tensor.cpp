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
} // namespace Weed
