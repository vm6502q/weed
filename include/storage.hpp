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

#include "common/weed_types.hpp"
#include "device_tag.hpp"
#include "dtype.hpp"

namespace Weed {
struct Storage;

typedef std::shared_ptr<Storage> StoragePtr;

struct Storage : public std::enable_shared_from_this<Storage> {
  DeviceTag device;
  DType dtype;
  vecCapInt size;

  Storage(DeviceTag dtg, DType dtp, vecCapInt n)
      : device(dtg), dtype(dtp), size(n) {
    if (!size) {
      throw std::invalid_argument("Storage must have size of at least 1!");
    }
  }

  virtual ~Storage() {}

  virtual StoragePtr get_ptr() { return shared_from_this(); }

  virtual int64_t get_device_id() { return -1; }

  virtual void FillZeros() = 0;
  virtual void FillOnes() = 0;

  virtual StoragePtr Upcast(DType dt) = 0;
};
} // namespace Weed
