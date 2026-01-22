//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
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

enum OCLAPI {
  OCL_API_UNKNOWN = 0,
  OCL_API_CLEAR_BUFFER_REAL = 1,
  OCL_API_FILL_ONES_REAL = 2,
  OCL_API_FILL_ONES_COMPLEX = 3,
  OCL_API_REAL_TO_COMPLEX_BUFFER = 4,
  OCL_API_RELU = 5,
  OCL_API_RELU_GRAD_REAL = 6,
  OCL_API_RELU_GRAD_COMPLEX = 7,
  OCL_API_ADD_REAL = 8,
  OCL_API_ADD_COMPLEX = 9,
  OCL_API_ADD_MIXED = 10,
  OCL_API_MUL_REAL = 11,
  OCL_API_MUL_COMPLEX = 12,
  OCL_API_MUL_MIXED = 13,
  OCL_API_MATMUL_REAL = 14,
  OCL_API_MATMUL_COMPLEX = 15,
  OCL_API_MATMUL_MIXED_C_LEFT = 16,
  OCL_API_MATMUL_MIXED_C_RIGHT = 17
};

} // namespace Weed
