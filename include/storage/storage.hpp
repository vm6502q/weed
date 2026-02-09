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
#include "enums/storage_type.hpp"

namespace Weed {
struct Storage;

typedef std::shared_ptr<Storage> StoragePtr;

/**
 * Base class for Tensor Storage for all data types
 */
struct Storage : public std::enable_shared_from_this<Storage> {
  /**
   * Type of storage
   */
  StorageType stype;
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

  Storage(const StorageType &stp, const DeviceTag &dtg, const DType &dtp,
          const tcapint &n)
      : stype(stp), device(dtg), dtype(dtp), size(n) {
    if (!size) {
      throw std::invalid_argument("Storage must have size of at least 1!");
    }
  }

  virtual ~Storage() {}

  /**
   * If sparse, return the sparse element count (or otherwise the dense size)
   */
  virtual tcapint get_sparse_size() const { return size; }

  /**
   * Is this Storage sparse?
   */
  virtual bool is_sparse() const { return false; }

  /**
   * Get a shared pointer to this Storage
   */
  virtual StoragePtr get_ptr() { return shared_from_this(); }

  /**
   * Get GPU device ID, if applicable
   */
  virtual int64_t get_device_id() const { return -1; }

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
  virtual StoragePtr Upcast(const DType &dt) = 0;

  /**
   * Migrate storage to CPU
   */
  virtual StoragePtr cpu() = 0;

  /**
   * Migrate storage to GPU
   */
  virtual StoragePtr gpu(const int64_t &did = -1) = 0;

  /**
   * Serialize storage to ostream
   */
  virtual void save(std::ostream &) const;

  /**
   * Load serialized storage from istream
   */
  static StoragePtr load(std::istream &);

  /**
   * Static helper function for storage serialization to ostream
   */
  static void write_storage_type(std::ostream &out, const StorageType &x) {
    out.write(reinterpret_cast<const char *>(&x), sizeof(StorageType));
  }

  /**
   * Static helper function for storage de-serialization from istream
   */
  static void read_storage_type(std::istream &in, StorageType &x) {
    in.read(reinterpret_cast<char *>(&x), sizeof(StorageType));
  }

  /**
   * Static helper function for storage serialization to ostream
   */
  static void write_device_tag(std::ostream &out, const DeviceTag &x) {
    out.write(reinterpret_cast<const char *>(&x), sizeof(DeviceTag));
  }

  /**
   * Static helper function for storage de-serialization from istream
   */
  static void read_device_tag(std::istream &in, DeviceTag &x) {
    in.read(reinterpret_cast<char *>(&x), sizeof(DeviceTag));
  }
};
} // namespace Weed
