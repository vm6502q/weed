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

#include "gpu_complex_storage.hpp"
#include "gpu_device.hpp"
#include "real_storage.hpp"

#if !ENABLE_OPENCL && !ENABLE_CUDA
#error GPU files were included without either OpenCL and CUDA enabled.
#endif

#include <list>

namespace Weed {
struct GpuRealStorage : RealStorage {
  GpuDevicePtr gpu;
  BufferPtr buffer;
  RealPtr array;

  GpuRealStorage(vecCapIntGpu n, int64_t did)
      : RealStorage(DeviceTag::GPU, n),
        gpu(OCLEngine::Instance().GetWeedDevice(did)),
        array(nullptr, [](real1 *) {}) {
    buffer = MakeBuffer(n);
  }

  ~GpuRealStorage() {}

  void FillZero() { gpu->ClearRealBuffer(buffer, size); }

  StoragePtr Upcast(DType dt) {
    if (dt == DType::REAL) {
      return get_ptr();
    }

    GpuComplexStorage n(size, gpu->deviceID);
    gpu->UpcastRealBuffer(buffer, n.buffer, size);

    return n.get_ptr();
  };

  BufferPtr MakeBuffer(vecCapIntGpu n) {
    if (gpu->device_context->use_host_mem) {
      array = Alloc(n);
      return gpu->MakeBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             sizeof(real1) * n, array.get());
    }

    return gpu->MakeBuffer(CL_MEM_READ_WRITE, sizeof(real1) * n);
  }
};
typedef std::shared_ptr<GpuRealStorage> GpuRealStoragePtr;
} // namespace Weed
