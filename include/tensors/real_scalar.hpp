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
/**
 * Tensor with only 1 element of real-number value (with broadcast on tensor
 * operations)
 *
 * No new properties or virtual methods are ever added beyond Weed::Tensor in
 * any of Weed::Scalar, Weed::RealScalar, or Weed::ComplexScalar, so it is
 * always possible (though not semantically "safe") to static_cast a
 * Weed::Tensor* based on its offset property to the scalar element to which the
 * Tensor.offset points, based on Tensor.storage->dtype. (Any addition of data
 * members, virtual methods, or multiple inheritance to these types or
 * sub-classes is a breaking change that violates this "unsafe" documented
 * feature.)
 */
struct RealScalar : public Scalar {
  RealScalar(const real1 &v, const bool &rg = false,
             DeviceTag dtag = DeviceTag::DEFAULT_DEVICE, int64_t did = -1)
      : Scalar(v, rg, dtag, did) {}

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
