//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#include "shared_api.hpp"

// "qfactory.hpp" pulls in all headers needed to create any type of
// "Qrack::QInterface."
#include "modules/module.hpp"
#include "storage/cpu_storage.hpp"
#include "tensors/symbol_tensor.hpp"

#include <fstream>
#include <iostream>
#include <mutex>

#define META_LOCK_GUARD()                                                      \
  const std::lock_guard<std::mutex> meta_lock(meta_operation_mutex)

#define MODULE_LOCK_GUARD(mid)                                                 \
  std::unique_ptr<const std::lock_guard<std::mutex>> module_lock;              \
  if (true) {                                                                  \
    std::lock(meta_operation_mutex, module_results[mid]->mtx);                 \
    const std::lock_guard<std::mutex> metaLock(meta_operation_mutex,           \
                                               std::adopt_lock);               \
    module_lock = std::make_unique<const std::lock_guard<std::mutex>>(         \
        module_results[mid]->mtx, std::adopt_lock);                            \
  }

#define MODULE_LOCK_GUARD_VOID(mid)                                            \
  MODULE_LOCK_GUARD(mid);                                                      \
  if ((mid >= module_results.size()) || !module_results[mid]) {                \
    std::cout << "Invalid argument: module ID not found!" << std::endl;        \
    meta_error = 2;                                                            \
    return;                                                                    \
  }

#define MODULE_LOCK_GUARD_INT(mid)                                             \
  MODULE_LOCK_GUARD(mid);                                                      \
  if ((mid >= module_results.size()) || !module_results[mid]) {                \
    std::cout << "Invalid argument: module ID not found!" << std::endl;        \
    meta_error = 2;                                                            \
    return 0U;                                                                 \
  }

using namespace Weed;

struct ModuleResult {
  std::mutex mtx;
  ModulePtr m;
  TensorPtr t;
  int error;
  ModuleResult(ModulePtr a) : m(a), t(nullptr), error(0) {}
};
typedef std::unique_ptr<ModuleResult> ModuleResultPtr;

std::mutex meta_operation_mutex;
int meta_error = 0;

std::vector<ModuleResultPtr> module_results;

extern "C" {
MICROSOFT_QUANTUM_DECL int get_error(_In_ const uintw mid) {
  if (meta_error) {
    meta_error = 0;
    return 2;
  }

  if ((mid >= module_results.size()) || !module_results[mid]) {
    std::cout << "Invalid argument: module ID not found!" << std::endl;
    return 2;
  }

  return module_results[mid]->error;
}

MICROSOFT_QUANTUM_DECL uintw load_module(_In_ const char *f) {
  META_LOCK_GUARD();

  bool is_success = true;
  ModulePtr m;
  try {
    std::ifstream i(f);
    m = Module::load(i);
    i.close();
    m->eval();
  } catch (const std::exception &ex) {
    std::cout << ex.what() << std::endl;
    is_success = false;
    meta_error = 1;
  }

  uintw id = 0U;
  if (is_success) {
    while ((id < module_results.size()) && module_results[id]) {
      ++id;
    }
    if (id == module_results.size()) {
      module_results.push_back(
          std::unique_ptr<ModuleResult>(new ModuleResult(m)));
    } else {
      module_results[id] = std::unique_ptr<ModuleResult>(new ModuleResult(m));
    }
  }

  return id;
}

MICROSOFT_QUANTUM_DECL void free_module(_In_ uintw mid) {
  MODULE_LOCK_GUARD_VOID(mid);
  module_results[mid] = nullptr;
}

MICROSOFT_QUANTUM_DECL void forward(_In_ uintw mid, _In_ uintw dtype,
                                    _In_ uintw n, _In_reads_(n) uintw *shape,
                                    _In_reads_(n) uintw *stride,
                                    _In_ double *d) {
  MODULE_LOCK_GUARD_VOID(mid);

  TensorPtr x;
  try {
    std::vector<tcapint> sh(n);
    std::vector<tcapint> st(n);
    for (size_t i = 0U; i < n; ++i) {
      sh[i] = (tcapint)shape[i];
      st[i] = (tcapint)stride[i];
    }

    tcapint max_index = 0U;
    for (size_t i = 0U; i < sh.size(); ++i) {
      max_index += (sh[i] - 1U) * st[i];
    }
    if (!sh.empty()) {
      ++max_index;
    }

    if (dtype == 1U) {
      std::vector<real1> v(max_index);
      for (size_t i = 0U; i < max_index; ++i) {
        v[i] = (real1)d[i];
      }
      x = std::make_shared<Tensor>(v, sh, st);
    } else {
      std::vector<complex> v(max_index);
      for (size_t i = 0U; i < max_index; ++i) {
        size_t j = i << 1U;
        v[i] = complex((real1)d[j], (real1)d[j + 1U]);
      }
      x = std::make_shared<Tensor>(v, sh, st);
    }
  } catch (const std::exception &ex) {
    std::cout << ex.what() << std::endl;
    meta_error = 2;
  }

  try {
    module_results[mid]->t = module_results[mid]->m->forward(x);
  } catch (const std::exception &ex) {
    std::cout << ex.what() << std::endl;
    module_results[mid]->error = 1;
  }
}

MICROSOFT_QUANTUM_DECL void forward_int(_In_ uintw mid, _In_ uintw dtype,
                                        _In_ uintw n,
                                        _In_reads_(n) uintw *shape,
                                        _In_reads_(n) uintw *stride,
                                        _In_ intw *d) {
  MODULE_LOCK_GUARD_VOID(mid);

  SymbolTensorPtr x;
  try {
    std::vector<tcapint> sh(n);
    std::vector<tcapint> st(n);
    for (size_t i = 0U; i < n; ++i) {
      sh[i] = (tcapint)shape[i];
      st[i] = (tcapint)stride[i];
    }

    tcapint max_index = 0U;
    for (size_t i = 0U; i < sh.size(); ++i) {
      max_index += (sh[i] - 1U) * st[i];
    }
    if (!sh.empty()) {
      ++max_index;
    }

    std::vector<symint> v(max_index);
    for (size_t i = 0U; i < max_index; ++i) {
      v[i] = (symint)d[i];
    }
    x = std::make_shared<SymbolTensor>(v, sh, st);
  } catch (const std::exception &ex) {
    std::cout << ex.what() << std::endl;
    meta_error = 2;
  }

  try {
    module_results[mid]->t = module_results[mid]->m->forward(x);
  } catch (const std::exception &ex) {
    std::cout << ex.what() << std::endl;
    module_results[mid]->error = 1;
  }
}

MICROSOFT_QUANTUM_DECL uintw get_result_index_count(_In_ uintw mid) {
  MODULE_LOCK_GUARD_INT(mid);

  const TensorPtr t = module_results[mid]->t;
  if (!t) {
    std::cout << "Invalid argument: module result tensor not found!"
              << std::endl;
    meta_error = 2;
    return 0U;
  }

  return (uintw)(t->shape.size());
}

MICROSOFT_QUANTUM_DECL void get_result_dims(_In_ uintw mid, uintw *shape,
                                            uintw *stride) {
  MODULE_LOCK_GUARD_VOID(mid);

  const TensorPtr t = module_results[mid]->t;
  if (!t) {
    std::cout << "Invalid argument: module result tensor not found!"
              << std::endl;
    meta_error = 2;
    return;
  }

  const size_t max_lcv = t->shape.size();
  for (size_t i = 0U; i < max_lcv; ++i) {
    shape[i] = t->shape[i];
    stride[i] = t->stride[i];
  }
}

MICROSOFT_QUANTUM_DECL uintw get_result_size(_In_ uintw mid) {
  MODULE_LOCK_GUARD_INT(mid);

  const TensorPtr t = module_results[mid]->t;
  if (!t) {
    std::cout << "Invalid argument: module result tensor not found!"
              << std::endl;
    meta_error = 2;
    return 0U;
  }

  return (uintw)(t->storage->size);
}

MICROSOFT_QUANTUM_DECL uintw get_result_offset(_In_ uintw mid) {
  MODULE_LOCK_GUARD_INT(mid);

  const TensorPtr t = module_results[mid]->t;
  if (!t) {
    std::cout << "Invalid argument: module result tensor not found!"
              << std::endl;
    meta_error = 2;
    return 0U;
  }

  return (uintw)(t->offset);
}

MICROSOFT_QUANTUM_DECL uintw get_result_type(_In_ uintw mid) {
  MODULE_LOCK_GUARD_INT(mid);

  const TensorPtr t = module_results[mid]->t;
  if (!t) {
    std::cout << "Invalid argument: module result tensor not found!"
              << std::endl;
    meta_error = 2;
    return 0U;
  }

  return (uintw)(t->storage->dtype);
}

MICROSOFT_QUANTUM_DECL void get_result(_In_ uintw mid, double *d) {
  MODULE_LOCK_GUARD_VOID(mid);

  const TensorPtr t = module_results[mid]->t;
  if (!t) {
    std::cout << "Invalid argument: module result tensor not found!"
              << std::endl;
    meta_error = 2;
    return;
  }

  const StoragePtr sp = t->storage->cpu();
  const size_t max_lcv = sp->size;
  if (sp->dtype == DType::COMPLEX) {
    ComplexStorage &s = *static_cast<ComplexStorage *>(sp.get());
    for (size_t i = 0U; i < max_lcv; ++i) {
      size_t j = i << 1U;
      const complex v = s[i];
      d[j] = (double)v.real();
      d[j + 1] = (double)v.imag();
    }
  } else {
    RealStorage &s = *static_cast<RealStorage *>(sp.get());
    for (size_t i = 0U; i < max_lcv; ++i) {
      d[i] = (double)s[i];
    }
  }
}
}
