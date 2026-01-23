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

#include <functional>

namespace Weed {
struct Node {
  std::vector<TensorPtr> parents;
  std::function<void()> backward;

  Node(std::vector<TensorPtr> p, std::function<void()> b)
      : parents(p), backward(b) {}
};
} // namespace Weed
