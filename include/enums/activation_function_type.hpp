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

#include "config.h"

namespace Weed {
/**
 * Activation function types for serialization
 */
enum ActivationFunctionType {
  NONE_FN = 0,
  SIGMOID_FN = 1,
  TANH_FN = 2,
  RELU_FN = 3,
  GELU_FN = 4,
  SWIGLU_FN = 5
};
} // namespace Weed
