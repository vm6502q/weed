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

#include "scalar.hpp"

#include "abs.hpp"
#include "add.hpp"
#include "matmul.hpp"
#include "mul.hpp"
#include "relu.hpp"

namespace Weed {
Scalar Scalar::abs(Scalar &a) {
  a.reset_indices();

  const bool rg = a.requires_grad();
  Scalar out = Scalar::allocate_like(a, a.storage->dtype, rg);

  Weed::abs(a, out);

  if (rg) {
    make_abs_node(a, out);
  }

  return out;
}

Scalar Scalar::relu(Scalar &a) {
  a.reset_indices();

  const bool rg = a.requires_grad();
  Scalar out = Scalar::allocate_like(a, a.storage->dtype, rg);

  Weed::relu(a, out);

  if (rg) {
    make_relu_node(a, out);
  }

  return out;
}

Scalar Scalar::add(Scalar &a, Scalar &b) {
  a.reset_indices();
  b.reset_indices();

  const bool rg = a.requires_grad() || b.requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  Scalar out = Scalar::allocate_like(a, dt, rg);

  Weed::add(a, b, out);

  if (rg) {
    make_add_node(a, b, out);
  }

  return out;
}

Scalar Scalar::mul(Scalar &a, Scalar &b) {
  a.reset_indices();
  b.reset_indices();

  const bool rg = a.requires_grad() || b.requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  Scalar out = Scalar::allocate_like(a, dt, rg);

  Weed::mul(a, b, out);

  if (rg) {
    make_mul_node(a, b, out);
  }

  return out;
}

Tensor Scalar::add(Scalar &a, Tensor &b) {
  a.match_shape(b);

  const bool rg = a.requires_grad() || b.requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  Tensor out = Scalar::allocate_like(a, dt, rg);

  Weed::add(a, b, out);

  if (rg) {
    make_add_node(a, b, out);
  }

  return out;
}

Tensor Scalar::mul(Scalar &a, Tensor &b) {
  a.match_shape(b);

  const bool rg = a.requires_grad() || b.requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  Tensor out = Scalar::allocate_like(a, dt, rg);

  Weed::mul(a, b, out);

  if (rg) {
    make_mul_node(a, b, out);
  }

  return out;
}
} // namespace Weed
