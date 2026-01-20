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
  OCL_API_ADD_REAL = 1,
  OCL_API_ADD_COMPLEX = 2,
  OCL_API_ADD_MIXED = 3,
  OCL_API_MUL_REAL = 4,
  OCL_API_MUL_COMPLEX = 5,
  OCL_API_MUL_MIXED = 6,
};

} // namespace Weed
