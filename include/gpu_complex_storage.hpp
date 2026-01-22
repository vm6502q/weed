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

#include "complex_storage.hpp"
#include "gpu_device.hpp"

#if !ENABLE_OPENCL && !ENABLE_CUDA
#error GPU files were included without either OpenCL and CUDA enabled.
#endif

#include <list>

namespace Weed {
struct GpuComplexStorage : ComplexStorage {
  GpuDevicePtr gpu;
  BufferPtr buffer;
  ComplexPtr array;

  GpuComplexStorage(vecCapIntGpu n, int64_t did)
      : ComplexStorage(DeviceTag::GPU, n),
        gpu(OCLEngine::Instance().GetWeedDevice(did)),
        array(nullptr, [](complex *) {}) {
    buffer = MakeBuffer(n);
  }

  ~GpuComplexStorage() {}

  void FillZeros() { gpu->ClearRealBuffer(buffer, size << 1U); }
  void FillOnes() { gpu->FillOnesComplex(buffer, size); }

  StoragePtr Upcast(DType dt) { return get_ptr(); };

  BufferPtr MakeBuffer(vecCapIntGpu n) {
    if (gpu->device_context->use_host_mem) {
      array = Alloc(n);
      return gpu->MakeBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             sizeof(complex) * n, array.get());
    }

    return gpu->MakeBuffer(CL_MEM_READ_WRITE, sizeof(complex) * n);
  }
};
} // namespace Weed
