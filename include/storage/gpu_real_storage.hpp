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

#include "devices/gpu_device.hpp"
#include "storage/gpu_complex_storage.hpp"
#include "storage/real_storage.hpp"

#if !ENABLE_OPENCL && !ENABLE_CUDA
#error GPU files were included without either OpenCL and CUDA enabled.
#endif

#include <list>

namespace Weed {
/**
 * GPU-accessible storage for real data type elements
 */
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

  GpuRealStorage(std::vector<real1> val, int64_t did)
      : RealStorage(DeviceTag::GPU, val.size()),
        gpu(OCLEngine::Instance().GetWeedDevice(did)),
        array(Alloc(val.size())) {
    std::copy(val.begin(), val.end(), array.get());
    buffer = MakeBuffer(val.size());
    array.reset();
  }

  ~GpuRealStorage() {}

  int64_t get_device_id() { return gpu->deviceID; }

  void FillZeros() { gpu->ClearRealBuffer(buffer, size); }
  void FillOnes() { gpu->FillOnesReal(buffer, size); }

  StoragePtr Upcast(DType dt) {
    if (dt == DType::REAL) {
      return get_ptr();
    }

    GpuComplexStoragePtr n =
        std::make_shared<GpuComplexStorage>(size, gpu->deviceID);
    gpu->UpcastRealBuffer(buffer, n->buffer, size);

    return n;
  };

  BufferPtr MakeBuffer(vecCapIntGpu n) {
    if (gpu->device_context->use_host_mem) {
      if (!array) {
        array = Alloc(n);
      }

      return gpu->MakeBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             sizeof(real1) * n, array.get());
    }

    if (!array) {
      return gpu->MakeBuffer(CL_MEM_READ_WRITE, sizeof(real1) * n);
    }

    return gpu->MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(real1) * n, array.get());
  }

  real1 operator[](vecCapInt idx) {
    if (idx >= size) {
      throw std::invalid_argument(
          "GpuRealStorage::operator[] argument out-of-bounds!");
    }

    return gpu->GetReal(buffer, (vecCapIntGpu)idx);
  }
};
typedef std::shared_ptr<GpuRealStorage> GpuRealStoragePtr;
} // namespace Weed
