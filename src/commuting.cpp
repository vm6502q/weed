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
    fn = [&](const vecCapIntGpu &i, const unsigned &cpu) { po[i] *= pa[i]; };  \
    break;                                                                     \
  case CommutingOperation::ADD:                                                \
  default:                                                                     \
    fn = [&](const vecCapIntGpu &i, const unsigned &cpu) { po[i] += pa[i]; };  \
  }                                                                            \
  size_t n = out->size;                                                        \
  pfControl.par_for(0, n, fn)

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
  void gpu_real(const Tensor &a, const Tensor &b, Tensor &out) {}
  void gpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {}
  void gpu_mixed(const Tensor &a, const Tensor &b, Tensor &out) {}

  void cpu_real_inplace(const StoragePtr a, StoragePtr out) {
    CAST_STORAGE(pa, a, real1, CpuRealStorage);
    CAST_STORAGE(po, out, real1, CpuRealStorage);

    KERNEL_SWITCH_INPLACE();
  }
  void cpu_complex_inplace(const StoragePtr a, StoragePtr out) {
    CAST_STORAGE(pa, a, complex, CpuComplexStorage);
    CAST_STORAGE(po, out, complex, CpuComplexStorage);

    KERNEL_SWITCH_INPLACE();
  }
  void gpu_real_inplace(const StoragePtr a, StoragePtr out) {}
  void gpu_complex_inplace(const StoragePtr a, StoragePtr out) {}
  void gpu_mixed_inplace(const StoragePtr a, StoragePtr out) {}
};
} // namespace Weed
