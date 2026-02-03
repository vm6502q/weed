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

#if !ENABLE_OPENCL && !ENABLE_CUDA
#error GPU files were included without either OpenCL and CUDA enabled.
#endif

namespace Weed {
/**
 * GPU-accessible storage
 */
struct GpuStorage {
  GpuDevicePtr dev;
  BufferPtr buffer;
  bool is_mapped;

  GpuStorage() : dev(nullptr), buffer(nullptr), is_mapped(false) {}

  void AddAlloc(const size_t &sz) { dev->AddAlloc(sz); }
  void SubtractAlloc(const size_t &sz) { dev->SubtractAlloc(sz); }

  BufferPtr MakeBuffer(const tcapint &n, const size_t &szf, void *array) {
    if (dev->device_context->use_host_mem) {
      return dev->MakeBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, szf * n,
                             array);
    }

    if (!array) {
      return dev->MakeBuffer(CL_MEM_READ_WRITE, szf * n);
    }

    return dev->MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, szf * n,
                           array);
  }
};
typedef std::shared_ptr<GpuStorage> GpuStoragePtr;
} // namespace Weed
