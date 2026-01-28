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

#define SPARSE_CPU_2_RUN(strg)                                                 \
  if (out.storage->is_sparse() && a.storage->is_sparse()) {                    \
    GET_STORAGE(strg, a, sa);                                                  \
    pfControl.par_for(sa->data, fn);                                           \
  } else {                                                                     \
    pfControl.par_for(0, n, fn);                                               \
  }

#define SPARSE_RUN(storage1, storage2)                                         \
  GET_STORAGE(storage1, a, sa);                                                \
  GET_STORAGE(storage2, b, sb);                                                \
  std::set<tcapint> keys;                                                      \
  for (auto it = sa->data.begin(); it != sa->data.end(); ++it) {               \
    keys.insert(it->first);                                                    \
  }                                                                            \
  for (auto it = sb->data.begin(); it != sb->data.end(); ++it) {               \
    keys.insert(it->first);                                                    \
  }                                                                            \
  pfControl.par_for(keys, fn);

#define SPARSE_CPU_2_SWITCH(storage1, storage2)                                \
  if (b.storage->is_sparse()) {                                                \
    GET_STORAGE(storage2, b, sb);                                              \
    pfControl.par_for(sb->data, fn);                                           \
  } else {                                                                     \
    pfControl.par_for(0, n, fn);                                               \
  }

#define SPARSE_CPU_3_RUN(storage1, storage2)                                   \
  if (a.storage->is_sparse() && b.storage->is_sparse()) {                      \
    GET_STORAGE(storage1, a, sa);                                              \
    GET_STORAGE(storage2, b, sb);                                              \
    std::set<tcapint> keys;                                                    \
    for (auto it = sa->data.begin(); it != sa->data.end(); ++it) {             \
      keys.insert(it->first);                                                  \
    }                                                                          \
    for (auto it = sb->data.begin(); it != sb->data.end(); ++it) {             \
      keys.insert(it->first);                                                  \
    }                                                                          \
    pfControl.par_for(keys, fn);                                               \
  } else {                                                                     \
    pfControl.par_for(0, n, fn);                                               \
  }

#define SPARSE_CPU_GRAD_3_RUN(storage1, storage2)                              \
  if (din.storage->is_sparse() && dout.storage->is_sparse()) {                 \
    GET_STORAGE(storage1, din, si);                                            \
    GET_STORAGE(storage2, dout, so);                                           \
    std::set<tcapint> keys;                                                    \
    for (auto it = si->data.begin(); it != si->data.end(); ++it) {             \
      keys.insert(it->first);                                                  \
    }                                                                          \
    for (auto it = so->data.begin(); it != so->data.end(); ++it) {             \
      keys.insert(it->first);                                                  \
    }                                                                          \
    pfControl.par_for(keys, fn);                                               \
  } else {                                                                     \
    pfControl.par_for(0, n, fn);                                               \
  }
