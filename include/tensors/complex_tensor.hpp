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
#include "tensors/tensor.hpp"

namespace Weed {
/**
 * Interface to read Tensor as flat and complex-valued
 *
 * No new properties or virtual methods are ever added beyond Weed::Tensor in
 * any sub-classes, so it is always possible (though not semantically "safe") to
 * static_cast a Weed::Tensor* based on its offset property to the scalar
 * element to which the Tensor.offset points, based on Tensor.storage->dtype.
 * (Any addition of data members, virtual methods, or multiple inheritance to
 * these types or sub-classes is a breaking change that violates this "unsafe"
 * documented feature.)
 */
struct ComplexTensor : public Tensor {
  ComplexTensor(const Tensor &orig) : Tensor(orig) {
    if (storage->dtype != DType::COMPLEX) {
      throw std::domain_error("ComplexTensor constructor must copy from a "
                              "complex-valued generic Tensor!");
    }
  }

  /**
   * Select element at flattened position
   */
  complex operator[](const tcapint &idx) const {
    return (*static_cast<TypedStorage<complex> *>(
        storage.get()))[get_storage_index(idx)];
  }

  /**
   * Set the real element at the position
   */
  void write(const tcapint &idx, const complex &val) {
    static_cast<TypedStorage<complex> *>(storage.get())
        ->write(get_storage_index(idx), val);
  }

  /**
   * Add to the real element at the position
   */
  void add(const tcapint &idx, const complex &val) {
    static_cast<TypedStorage<complex> *>(storage.get())
        ->add(get_storage_index(idx), val);
  }
};

typedef std::shared_ptr<ComplexTensor> ComplexTensorPtr;
} // namespace Weed
