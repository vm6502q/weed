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
 * Quantum function types for serialization
 */
enum QuantumFunctionType { CUSTOM_QFN = 0, NONE_QFN = 1, BELL_GHZ_QFN = 2 };
} // namespace Weed
