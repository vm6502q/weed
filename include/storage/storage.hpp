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
#include "enums/device_tag.hpp"
#include "enums/dtype.hpp"

namespace Weed {
struct Storage;

typedef std::shared_ptr<Storage> StoragePtr;

/**
 * Base class for Tensor Storage for all data types
 */
struct Storage : public std::enable_shared_from_this<Storage> {
  /**
   * GPU device ID, if applicable
   */
  DeviceTag device;
  /**
   * Data type of this storage
   */
  DType dtype;
  /**
   * Number of elements (of data type) in this storage
   */
  tcapint size;

  Storage(DeviceTag dtg, DType dtp, tcapint n)
      : device(dtg), dtype(dtp), size(n) {
    if (!size) {
      throw std::invalid_argument("Storage must have size of at least 1!");
    }
  }

  /**
   * Get a shared pointer to this Storage
   */
  virtual StoragePtr get_ptr() { return shared_from_this(); }

  /**
   * Get GPU device ID, if applicable
   */
  virtual int64_t get_device_id() { return -1; }

  /**
   * Fill the entire Storage with 0 values
   */
  virtual void FillZeros() = 0;
  /**
   * Fill the entire Storage with 1 values
   */
  virtual void FillOnes() = 0;

  /**
   * If this is real-number storage, up-cast to complex-number storage (by
   * doubling stride)
   */
  virtual StoragePtr Upcast(DType dt) = 0;

  /**
   * Migrate storage to CPU
   */
  virtual StoragePtr cpu() = 0;

  /**
   * Migrate storage to GPU
   */
  virtual StoragePtr gpu(int64_t did = -1) = 0;
};
} // namespace Weed
