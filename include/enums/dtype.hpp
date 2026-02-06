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

namespace Weed {
/**
 * Data types available in Weed
 */
enum DType {
  NONE_DTYPE = 0,
  REAL = 1,
  COMPLEX = 2,
  INT = 3,
  DEFAULT_DTYPE = REAL
};
} // namespace Weed
