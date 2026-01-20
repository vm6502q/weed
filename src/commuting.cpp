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

#define CAST_STORAGE(out, in, type, ptr) type *out = static_cast<ptr*>(in.storage.get())->data.get() + in.offset

namespace Weed {
ParallelFor pfControl = ParallelFor();

struct commuting_kernel : CommutingKernel {
  void cpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
    CAST_STORAGE(pa, a, real1, CpuRealStorage);
    CAST_STORAGE(pb, b, real1, CpuRealStorage);
    CAST_STORAGE(po, out, real1, CpuRealStorage);

    size_t n = out.storage->size;
 
    ParallelFunc fn;
    switch (op) {
      case CommutingOperation::MUL:
        fn = [&](const vecCapIntGpu &i, const unsigned &cpu) {
          po[i] = pa[i] * pb[i];
        };
        break;
      case CommutingOperation::ADD:
      default:
        fn = [&](const vecCapIntGpu &i, const unsigned &cpu) {
          po[i] = pa[i] + pb[i];
        };
    }

    pfControl.par_for(0, n, fn);
  }
  void cpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
    CAST_STORAGE(pa, a, complex, CpuComplexStorage);
    CAST_STORAGE(pb, b, complex, CpuComplexStorage);
    CAST_STORAGE(po, out, complex, CpuComplexStorage);

    size_t n = out.storage->size;

    ParallelFunc fn;
    switch (op) {
      case CommutingOperation::MUL:
        fn = [&](const vecCapIntGpu &i, const unsigned &cpu) {
          po[i] = pa[i] * pb[i];
        };
        break;
      case CommutingOperation::ADD:
      default:
        fn = [&](const vecCapIntGpu &i, const unsigned &cpu) {
          po[i] = pa[i] + pb[i];
        };
    }

    pfControl.par_for(0, n, fn);
  }
  void cpu_mixed(const Tensor &a, const Tensor &b, Tensor &out) {
    CAST_STORAGE(pa, a, complex, CpuComplexStorage);
    CAST_STORAGE(pb, b, real1, CpuRealStorage);
    CAST_STORAGE(po, out, complex, CpuComplexStorage);

    size_t n = out.storage->size;

    ParallelFunc fn;
    switch (op) {
      case CommutingOperation::MUL:
        fn = [&](const vecCapIntGpu &i, const unsigned &cpu) {
          po[i] = pa[i] * pb[i];
        };
        break;
      case CommutingOperation::ADD:
      default:
        fn = [&](const vecCapIntGpu &i, const unsigned &cpu) {
          po[i] = pa[i] + pb[i];
        };
    }

    pfControl.par_for(0, n, fn);
  }
  void cpu_promote(const Tensor &a, const Tensor &b, Tensor &out) {
    CAST_STORAGE(pa, a, real1, CpuRealStorage);
    CAST_STORAGE(pb, b, real1, CpuRealStorage);
    CAST_STORAGE(po, out, complex, CpuComplexStorage);

    size_t n = out.storage->size;

    ParallelFunc fn;
    switch (op) {
      case CommutingOperation::MUL:
        fn = [&](const vecCapIntGpu &i, const unsigned &cpu) {
          po[i] = pa[i] * pb[i];
        };
        break;
      case CommutingOperation::ADD:
      default:
        fn = [&](const vecCapIntGpu &i, const unsigned &cpu) {
          po[i] = pa[i] + pb[i];
        };
    }

    pfControl.par_for(0, n, fn);
  }
  void opencl_real(const Tensor &a, const Tensor &b, Tensor &out) {}
  void opencl_complex(const Tensor &a, const Tensor &b, Tensor &out) {}
  void opencl_mixed(const Tensor &a, const Tensor &b, Tensor &out) {}
  void opencl_promote(const Tensor &a, const Tensor &b, Tensor &out) {}
};
} // namespace Weed
