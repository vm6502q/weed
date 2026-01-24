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
#include "storage/complex_storage.hpp"

namespace Weed {
struct ComplexScalar : public Scalar {
  ComplexScalar(complex v, bool rg = false, DeviceTag dtag = DeviceTag::CPU,
                int64_t did = -1)
      : Scalar(v, rg, dtag, did) {}
  ComplexScalar(TensorPtr orig) : Scalar(orig) {
    if (orig->storage->dtype != DType::COMPLEX) {
      throw std::invalid_argument(
          "Cannot construct ComplexScalar from non-complex Tensor!");
    }
  }

  complex get_item() const {
    return (*static_cast<ComplexStorage *>(storage.get()))[offset];
  }
};

typedef std::shared_ptr<ComplexScalar> ComplexScalarPtr;

inline bool operator==(const ComplexScalar &left, const ComplexScalar &right) {
  return left.get_item() == right.get_item();
}
inline bool operator!=(const ComplexScalar &left, const ComplexScalar &right) {
  return left.get_item() != right.get_item();
}
} // namespace Weed
