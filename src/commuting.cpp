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

#include "commuting.hpp"
#include "common/parallel_for.hpp"
#include "cpu_complex_storage.hpp"
#include "cpu_real_storage.hpp"
#include "gpu_complex_storage.hpp"
#include "gpu_real_storage.hpp"

#define CAST_TENOSR_STORAGE(out, in, type, ptr)                                \
  type *out = static_cast<ptr *>(in.storage.get())->data.get() + in.offset

#define CAST_STORAGE(out, in, type, ptr)                                       \
  type *out = static_cast<ptr *>(in.get())->data.get()

#define KERNEL_SWITCH()                                                        \
  ParallelFunc fn;                                                             \
  switch (op) {                                                                \
  case CommutingOperation::MUL:                                                \
    fn = [&](const vecCapIntGpu &i, const unsigned &cpu) {                     \
      po[i] = pa[i] * pb[i];                                                   \
    };                                                                         \
    break;                                                                     \
  case CommutingOperation::ADD:                                                \
  default:                                                                     \
    fn = [&](const vecCapIntGpu &i, const unsigned &cpu) {                     \
      po[i] = pa[i] + pb[i];                                                   \
    };                                                                         \
  }                                                                            \
  size_t n = out.storage->size;                                                \
  pfControl.par_for(0, n, fn)

#define KERNEL_SWITCH_INPLACE()                                                \
  ParallelFunc fn;                                                             \
  switch (op) {                                                                \
  case CommutingOperation::MUL:                                                \
    fn = [&](const vecCapIntGpu &i, const unsigned &cpu) { pa[i] *= pb[i]; };  \
    break;                                                                     \
  case CommutingOperation::ADD:                                                \
  default:                                                                     \
    fn = [&](const vecCapIntGpu &i, const unsigned &cpu) { pa[i] += pb[i]; };  \
  }                                                                            \
  size_t n = b->size;                                                          \
  pfControl.par_for(0, n, fn)

#define DISPATCH_GPU_KERNEL(type, type2, api_add, api_mul)                     \
  OCLAPI api_call;                                                             \
  switch (op) {                                                                \
  case CommutingOperation::MUL:                                                \
    api_call = api_mul;                                                        \
    break;                                                                     \
  case CommutingOperation::ADD:                                                \
  default:                                                                     \
    api_call = api_add;                                                        \
  }                                                                            \
  const vecCapIntGpu args[2U]{a.offset, b.offset};                             \
  std::shared_ptr<type> a_storage =                                            \
      std::dynamic_pointer_cast<type>(a.storage);                              \
  std::shared_ptr<type2> b_storage =                                           \
      std::dynamic_pointer_cast<type2>(b.storage);                             \
  std::shared_ptr<type> o_storage =                                            \
      std::dynamic_pointer_cast<type>(out.storage);                            \
  a_storage->gpu->RequestKernel(                                               \
      api_call, args, a.get_size(),                                            \
      {a_storage->buffer, b_storage->buffer, o_storage->buffer})

namespace Weed {
ParallelFor pfControl = ParallelFor();

struct commuting_kernel : CommutingKernel {
  void cpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
    CAST_TENOSR_STORAGE(pa, a, real1, CpuRealStorage);
    CAST_TENOSR_STORAGE(pb, b, real1, CpuRealStorage);
    CAST_TENOSR_STORAGE(po, out, real1, CpuRealStorage);

    KERNEL_SWITCH();
  }
  void cpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
    CAST_TENOSR_STORAGE(pa, a, complex, CpuComplexStorage);
    CAST_TENOSR_STORAGE(pb, b, complex, CpuComplexStorage);
    CAST_TENOSR_STORAGE(po, out, complex, CpuComplexStorage);

    KERNEL_SWITCH();
  }
  void cpu_mixed(const Tensor &a, const Tensor &b, Tensor &out) {
    CAST_TENOSR_STORAGE(pa, a, complex, CpuComplexStorage);
    CAST_TENOSR_STORAGE(pb, b, real1, CpuRealStorage);
    CAST_TENOSR_STORAGE(po, out, complex, CpuComplexStorage);

    KERNEL_SWITCH();
  }
  void gpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
    DISPATCH_GPU_KERNEL(GpuRealStorage, GpuRealStorage,
                        OCLAPI::OCL_API_ADD_REAL, OCLAPI::OCL_API_MUL_REAL);
  }
  void gpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
    DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuComplexStorage,
                        OCLAPI::OCL_API_ADD_COMPLEX,
                        OCLAPI::OCL_API_MUL_COMPLEX);
  }
  void gpu_mixed(const Tensor &a, const Tensor &b, Tensor &out) {
    DISPATCH_GPU_KERNEL(GpuComplexStorage, GpuRealStorage,
                        OCLAPI::OCL_API_ADD_MIXED, OCLAPI::OCL_API_MUL_MIXED);
  }

  void cpu_real_inplace(StoragePtr a, const StoragePtr b) {
    CAST_STORAGE(pa, a, real1, CpuRealStorage);
    CAST_STORAGE(pb, b, real1, CpuRealStorage);

    KERNEL_SWITCH_INPLACE();
  }
  void cpu_complex_inplace(StoragePtr a, const StoragePtr b) {
    CAST_STORAGE(pa, a, complex, CpuComplexStorage);
    CAST_STORAGE(pb, b, complex, CpuComplexStorage);

    KERNEL_SWITCH_INPLACE();
  }
  void gpu_real_inplace(StoragePtr a, const StoragePtr b) {}
  void gpu_complex_inplace(StoragePtr a, const StoragePtr b) {}
  void gpu_mixed_inplace(StoragePtr a, const StoragePtr b) {}
};
} // namespace Weed
