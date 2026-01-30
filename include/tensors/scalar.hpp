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

#include "tensor.hpp"

namespace Weed {
struct Scalar;

typedef std::shared_ptr<Scalar> ScalarPtr;

/**
 * Tensor with only 1 element (with broadcast on tensor operations)
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
struct Scalar : public Tensor {
  Scalar(const real1 &v, const bool &rg,
         const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
         const int64_t &did = -1)
      : Tensor(v, rg, dtag, did) {}
  Scalar(const complex &v, const bool &rg,
         const DeviceTag &dtag = DeviceTag::DEFAULT_DEVICE,
         const int64_t &did = -1)
      : Tensor(v, rg, dtag, did) {}
};
} // namespace Weed
