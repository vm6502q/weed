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
  LSTM_T = 10,
  MIGRATE_CPU_T = 11,
  MIGRATE_GPU_T = 12,
  SOFTMAX_T = 13,
  LOGSOFTMAX_T = 14,
  QRACK_NEURON_T = 15,
  QRACK_NEURON_LAYER_T = 16,
  MULTIHEAD_ATTENTION_T = 17,
  TRANSFORMER_ENCODER_LAYER_T = 18,
  GELU_T = 19,
  MEAN_T = 20,
  MIN_T = 21,
  MAX_T = 22,
  RESHAPE_T = 23,
  VARIANCE_T = 24,
  STDDEV_T = 25,
  POSITIONAL_ENCODING_T = 26,
  MEAN_CENTER_T = 27
};
} // namespace Weed
