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

#include "storage/typed_storage.hpp"

namespace Weed {
/**
 * Storage for real data type elements
 */
struct ComplexStorage : TypedStorage<complex> {
  ComplexStorage(const DeviceTag &dtg, const tcapint &n)
      : TypedStorage<complex>(dtg, n) {}
};
typedef std::shared_ptr<ComplexStorage> ComplexStoragePtr;
} // namespace Weed
