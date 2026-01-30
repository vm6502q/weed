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

#include "tensors/tensor.hpp"

#include <functional>

namespace Weed {
/**
 * Autograd graph node
 */
struct Node {
  /**
   * Parent tensors of this operation in the autograd graph
   */
  std::vector<TensorPtr> parents;
  /**
   * Autograd back-propagation function
   */
  std::function<void()> backward;

  /**
   * Used by Weed::Tensor or user code to construct an autograd graph node
   */
  Node(const std::vector<TensorPtr> &p, const std::function<void()> &b)
      : parents(p), backward(b) {
    for (auto &t : parents) {
      t->make_gradient();
    }
  }
};
} // namespace Weed
