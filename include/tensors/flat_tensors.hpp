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

#include "storage/all_storage.hpp"
#include "tensors/complex_tensor.hpp"
#include "tensors/real_tensor.hpp"

#define GET_FLAT_TENSOR(type, i, o) type *o = static_cast<type *>(&i);

#define GET_CONST_FLAT_TENSOR(type, i, o) GET_FLAT_TENSOR(const type, i, o)

#define CPU_INIT_2_SCALAR(ft, strg)                                            \
  GET_CONST_FLAT_TENSOR(ft, a, pa);                                            \
  GET_FLAT_TENSOR(strg, out, po);                                              \
  const size_t n = a.get_broadcast_size()

#define CPU_INIT_2(ft, strg)                                                   \
  GET_CONST_FLAT_TENSOR(ft, a, pa);                                            \
  GET_FLAT_TENSOR(strg, out, po);                                              \
  const size_t n = out.storage->size

#define CPU_INIT_2_IN_PLACE(ft1, ft2)                                          \
  GET_FLAT_TENSOR(ft1, a, pa);                                                 \
  GET_CONST_FLAT_TENSOR(ft2, b, pb);                                           \
  const size_t n = a.get_broadcast_size()

#define CPU_INIT_3(ft1, ft2, strg)                                             \
  GET_CONST_FLAT_TENSOR(ft1, a, pa);                                           \
  GET_CONST_FLAT_TENSOR(ft2, b, pb);                                           \
  GET_FLAT_TENSOR(strg, out, po);                                              \
  const size_t n = out.storage->size

#define CPU_GRAD_INIT_3(ft1, ft2, ft3)                                         \
  GET_FLAT_TENSOR(ft1, din, pdi);                                              \
  GET_CONST_FLAT_TENSOR(ft2, in, pi);                                          \
  GET_CONST_FLAT_TENSOR(ft3, dout, po);                                        \
  const size_t n = din.get_broadcast_size()
