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

#define GPU_INIT_2_SCALAR(storage1, storage2)                                  \
  GET_STORAGE(storage1, a, pa);                                                \
  size_t n = a.get_broadcast_size()

#define SPARSE_CPU_2_RUN(strg)                                                 \
  if (out.storage->is_sparse() && a.storage->is_sparse() &&                    \
      a.is_contiguous()) {                                                     \
    GET_STORAGE(strg, a, sa);                                                  \
    pfControl.par_for(sa->data, fn);                                           \
  } else {                                                                     \
    pfControl.par_for(0, n, fn);                                               \
  }

#define SPARSE_CPU_2_SWITCH(strg)                                              \
  if (b.storage->is_sparse() && b.is_contiguous()) {                           \
    GET_STORAGE(strg, b, sb);                                                  \
    pfControl.par_for(sb->data, fn);                                           \
  } else {                                                                     \
    pfControl.par_for(0, n, fn);                                               \
  }

#define SPARSE_CPU_3_RUN(storage1, storage2)                                   \
  if (out.storage->is_sparse() && a.storage->is_sparse() &&                    \
      b.storage->is_sparse() && a.is_contiguous() && b.is_contiguous()) {      \
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
  if (din.storage->is_sparse() && dout.storage->is_sparse() &&                 \
      din.is_contiguous() && dout.is_contiguous()) {                           \
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
