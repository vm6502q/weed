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
 * Storage types for serialization
 */
enum StorageType {
  NONE_STORAGE_TYPE = 0,
  REAL_CPU_DENSE = 1,
  REAL_GPU_DENSE = 2,
  COMPLEX_CPU_DENSE = 3,
  COMPLEX_GPU_DENSE = 4,
  INT_CPU_DENSE = 5,
  INT_GPU_DENSE = 6,
  REAL_CPU_SPARSE = 7,
  COMPLEX_CPU_SPARSE = 8
};
} // namespace Weed
