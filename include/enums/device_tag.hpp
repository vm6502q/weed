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
enum DeviceTag {
  CPU = 1,
  GPU = 2,
  Qrack = 3,
#if ENABLE_OPENCL || ENABLE_CUDA
  DEFAULT_DEVICE = GPU
#else
  DEFAULT_DEVICE = CPU
#endif
};
} // namespace Weed
