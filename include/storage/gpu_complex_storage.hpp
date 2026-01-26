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
#include "storage/complex_storage.hpp"
#include "storage/gpu_storage.hpp"

#if !ENABLE_OPENCL && !ENABLE_CUDA
#error GPU files were included without either OpenCL and CUDA enabled.
#endif

#include <list>

namespace Weed {
/**
 * GPU-accessible storage for complex data type elements
 */
struct GpuComplexStorage : public ComplexStorage, public GpuStorage {
  ComplexPtr array;

  GpuComplexStorage(vecCapIntGpu n, int64_t did)
      : ComplexStorage(DeviceTag::GPU, n),
        array(nullptr, [](complex *) {}) {
    gpu = OCLEngine::Instance().GetWeedDevice(did);
    AddAlloc(sizeof(complex) * size);
    buffer = MakeBuffer(n);
  }

  GpuComplexStorage(std::vector<complex> val, int64_t did)
      : ComplexStorage(DeviceTag::GPU, val.size()),
        array(Alloc(val.size())) {
    gpu = OCLEngine::Instance().GetWeedDevice(did);
    AddAlloc(sizeof(complex) * size);
    std::copy(val.begin(), val.end(), array.get());
    buffer = MakeBuffer(val.size());
    array.reset();
  }

  virtual ~GpuComplexStorage() { SubtractAlloc(sizeof(complex) * size); }

  void FillZeros() override { gpu->ClearRealBuffer(buffer, size << 1U); }
  void FillOnes() override { gpu->FillOnesComplex(buffer, size); }
  void FillValue(complex v) override { gpu->FillValueComplex(buffer, size, v); }

  StoragePtr Upcast(DType dt) { return get_ptr(); };

  BufferPtr MakeBuffer(vecCapIntGpu n) {
    if (gpu->device_context->use_host_mem) {
      if (!array) {
        array = Alloc(n);
      }

      return gpu->MakeBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             sizeof(complex) * n, array.get());
    }

    if (!array) {
      return gpu->MakeBuffer(CL_MEM_READ_WRITE, sizeof(complex) * n);
    }

    return gpu->MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(complex) * n, array.get());
  }

  complex operator[](vecCapInt idx) {
    if (idx >= size) {
      throw std::invalid_argument(
          "GpuComplexStorage::operator[] argument out-of-bounds!");
    }

    vecCapIntGpu i = ((vecCapIntGpu)idx) << 1U;

    return complex(gpu->GetReal(buffer, i), gpu->GetReal(buffer, i + 1U));
  }
};
typedef std::shared_ptr<GpuComplexStorage> GpuComplexStoragePtr;
} // namespace Weed
