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

#include "add.hpp"
#include "common/parallel_for.hpp"
#include "cpu_complex_storage.hpp"
#include "cpu_real_storage.hpp"

namespace Weed {
ParallelFor pfControl = ParallelFor();

struct add_kernel {
  void cpu_real(const Tensor &a, const Tensor &b, Tensor &out) {
    real1 *pa =
        static_cast<CpuRealStorage *>(a.storage.get())->data.get() + a.offset;
    real1 *pb =
        static_cast<CpuRealStorage *>(b.storage.get())->data.get() + b.offset;
    real1 *po = static_cast<CpuRealStorage *>(out.storage.get())->data.get() +
                out.offset;

    size_t n = out.storage->size;

    pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
      po[i] = pa[i] + pb[i];
    });
  }
  void cpu_complex(const Tensor &a, const Tensor &b, Tensor &out) {
    complex *pa = static_cast<CpuComplexStorage *>(a.storage.get())->data.get() +
                a.offset;
    complex *pb = static_cast<CpuComplexStorage *>(b.storage.get())->data.get() +
                b.offset;
    complex *po =
        static_cast<CpuComplexStorage *>(out.storage.get())->data.get() +
        out.offset;

    size_t n = out.storage->size;

    pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
      po[i] = pa[i] + pb[i];
    });
  }
  void cpu_mixed(const Tensor &a, const Tensor &b, Tensor &out) {
    complex *pa = static_cast<CpuComplexStorage *>(a.storage.get())->data.get() +
                a.offset;
    real1 *pb = static_cast<CpuRealStorage *>(b.storage.get())->data.get() +
                b.offset;
    complex *po =
        static_cast<CpuComplexStorage *>(out.storage.get())->data.get() +
        out.offset;

    size_t n = out.storage->size;

    pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
      po[i] = pa[i] + pb[i];
    });
  }
  void cpu_promote(const Tensor &a, const Tensor &b, Tensor &out) {
    real1 *pa = static_cast<CpuRealStorage *>(a.storage.get())->data.get() +
                a.offset;
    real1 *pb = static_cast<CpuRealStorage *>(b.storage.get())->data.get() +
                b.offset;
    complex *po =
        static_cast<CpuComplexStorage *>(out.storage.get())->data.get() +
        out.offset;

    size_t n = out.storage->size;

    pfControl.par_for(0, n, [&](const vecCapIntGpu &i, const unsigned &cpu) {
      po[i] = pa[i] + pb[i];
    });
  }
  void opencl_real(const Tensor &a, const Tensor &b, Tensor &out) {}
  void opencl_complex(const Tensor &a, const Tensor &b, Tensor &out) {}
  void opencl_mixed(const Tensor &a, const Tensor &b, Tensor &out) {}
  void opencl_promote(const Tensor &a, const Tensor &b, Tensor &out) {}
};
} // namespace Weed
