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
#include "storage/cpu_complex_storage.hpp"
#include "storage/cpu_real_storage.hpp"
#include "storage/sparse_cpu_complex_storage.hpp"
#include "storage/sparse_cpu_real_storage.hpp"
#if ENABLE_GPU
#include "storage/gpu_complex_storage.hpp"
#include "storage/gpu_real_storage.hpp"
#endif

#define GET_STORAGE(type, i, o) type *o = static_cast<type *>(i.storage.get());

#define CPU_INIT_2_SCALAR(storage1, storage2)                                  \
  const tcapint O_a = a.offset;                                                \
  const tcapint I_a = a.stride[0U];                                            \
  GET_STORAGE(storage1, a, pa);                                                \
  GET_STORAGE(storage2, out, po);                                              \
  size_t n = a.get_size()

#define GPU_INIT_2_SCALAR(storage1, storage2)                                  \
  const tcapint O_a = a.offset;                                                \
  const tcapint I_a = a.stride[0U];                                            \
  GET_STORAGE(storage1, a, pa);                                                \
  size_t n = a.get_size()

#define CPU_INIT_2(storage1, storage2)                                         \
  const tcapint O_a = a.offset;                                                \
  const tcapint I_a = a.stride[0U];                                            \
  const tcapint I_o = out.stride[0U];                                          \
  GET_STORAGE(storage1, a, pa);                                                \
  GET_STORAGE(storage2, out, po);                                              \
  size_t n = out.storage->size

#define CPU_INIT_2_IN_PLACE(storage1, storage2)                                \
  const tcapint O_a = a.offset;                                                \
  const tcapint I_a = a.stride[0U];                                            \
  const tcapint O_b = b.offset;                                                \
  const tcapint I_b = b.stride[0U];                                            \
  GET_STORAGE(storage1, a, pa);                                                \
  GET_STORAGE(storage2, b, pb);                                                \
  size_t n = b.storage->size

#define CPU_INIT_3(storage1, storage2, storage3)                               \
  const tcapint O_a = a.offset;                                                \
  const tcapint I_a = a.stride[0U];                                            \
  const tcapint O_b = b.offset;                                                \
  const tcapint I_b = b.stride[0U];                                            \
  const tcapint I_o = out.stride[0U];                                          \
  GET_STORAGE(storage1, a, pa);                                                \
  GET_STORAGE(storage2, b, pb);                                                \
  GET_STORAGE(storage3, out, po);                                              \
  size_t n = out.storage->size

#define CPU_GRAD_INIT_3(storage1, storage2, storage3)                          \
  const tcapint O_d = din.offset;                                              \
  const tcapint I_d = din.stride[0U];                                          \
  const tcapint O_i = in.offset;                                               \
  const tcapint I_i = in.stride[0U];                                           \
  const tcapint O_o = dout.offset;                                             \
  const tcapint I_o = dout.stride[0U];                                         \
  GET_STORAGE(storage1, din, pdi);                                             \
  GET_STORAGE(storage2, in, pi);                                               \
  GET_STORAGE(storage3, dout, po);                                             \
  size_t n = dout.storage->size
