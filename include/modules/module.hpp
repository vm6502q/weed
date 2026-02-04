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

#include "tensors/parameter.hpp"

namespace Weed {
/**
 * Composable module with forward function and parameters for autograd
 * optimization
 */
struct Module {
  virtual TensorPtr forward(const TensorPtr) = 0;
  virtual std::vector<ParameterPtr> parameters() {
    return std::vector<ParameterPtr>();
  }
  virtual void train() {}
  virtual void eval() {}
  virtual ~Module() {}
};
typedef std::shared_ptr<Module> ModulePtr;
} // namespace Weed
