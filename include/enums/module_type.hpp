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
 * Module types for serialization
 */
enum ModuleType {
  NONE_MODULE_TYPE = 0,
  SEQUENTIAL_T = 1,
  LINEAR_T = 2,
  RELU_T = 3,
  SIGMOID_T = 4,
  TANH_T = 5,
  DROPOUT_T = 6,
  LAYERNORM_T = 7,
  EMBEDDING_T = 8,
  GRU_T = 9,
  LSTM_T = 10
};
} // namespace Weed
