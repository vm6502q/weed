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

#include "scalar.hpp"
#include "storage/real_storage.hpp"

namespace Weed {
struct RealScalar : public Scalar {
  RealScalar(real1 v, bool rg = false, DeviceTag dtag = DeviceTag::CPU,
             int64_t did = -1)
      : Scalar(v, rg, dtag, did) {}
  RealScalar(TensorPtr orig) : Scalar(orig) {
    if (orig->storage->dtype != DType::REAL) {
      throw std::invalid_argument(
          "Cannot construct RealScalar from non-real Tensor!");
    }
  }

  real1 get_item() const {
    return (*static_cast<RealStorage *>(storage.get()))[offset];
  }
};

typedef std::shared_ptr<RealScalar> RealScalarPtr;

inline bool operator==(const RealScalar &left, const RealScalar &right) {
  return left.get_item() == right.get_item();
}
inline bool operator!=(const RealScalar &left, const RealScalar &right) {
  return left.get_item() != right.get_item();
}
inline bool operator<(const RealScalar &left, const RealScalar &right) {
  return left.get_item() < right.get_item();
}
inline bool operator<=(const RealScalar &left, const RealScalar &right) {
  return left.get_item() <= right.get_item();
}
inline bool operator>(const RealScalar &left, const RealScalar &right) {
  return left.get_item() > right.get_item();
}
inline bool operator>=(const RealScalar &left, const RealScalar &right) {
  return left.get_item() >= right.get_item();
}
} // namespace Weed
