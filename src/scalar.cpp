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
ScalarPtr Scalar::abs(ScalarPtr a) {
  a->reset_indices();

  const bool rg = a->requires_grad();
  ScalarPtr out = Scalar::allocate_like(a, a->storage->dtype, rg);

  Weed::abs(*(a.get()), *(out.get()));

  if (rg) {
    make_abs_node(a, out);
  }

  return out;
}

ScalarPtr Scalar::relu(ScalarPtr a) {
  a->reset_indices();

  const bool rg = a->requires_grad();
  ScalarPtr out = Scalar::allocate_like(a, a->storage->dtype, rg);

  Weed::relu(*(a.get()), *(out.get()));

  if (rg) {
    make_relu_node(a, out);
  }

  return out;
}

ScalarPtr Scalar::add(ScalarPtr a, ScalarPtr b) {
  a->reset_indices();
  b->reset_indices();

  const bool rg = a->requires_grad() || b->requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  ScalarPtr out = Scalar::allocate_like(a, dt, rg);

  Weed::add(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_add_node(a, b, out);
  }

  return out;
}

ScalarPtr Scalar::mul(ScalarPtr a, ScalarPtr b) {
  a->reset_indices();
  b->reset_indices();

  const bool rg = a->requires_grad() || b->requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  ScalarPtr out = Scalar::allocate_like(a, dt, rg);

  Weed::mul(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_mul_node(a, b, out);
  }

  return out;
}

TensorPtr Scalar::add(ScalarPtr a, TensorPtr b) {
  a->match_shape(b);

  const bool rg = a->requires_grad() || b->requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  TensorPtr out = Scalar::allocate_like(a, dt, rg);

  Weed::add(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_add_node(a, b, out);
  }

  return out;
}

TensorPtr Scalar::mul(ScalarPtr a, TensorPtr b) {
  a->match_shape(b);

  const bool rg = a->requires_grad() || b->requires_grad();
  DType dt = get_dtype_by_presidence(a, b);
  TensorPtr out = Scalar::allocate_like(a, dt, rg);

  Weed::mul(*(a.get()), *(b.get()), *(out.get()));

  if (rg) {
    make_mul_node(a, b, out);
  }

  return out;
}
} // namespace Weed
