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

namespace Weed {
/**
 * GPU-accessible storage for complex data type elements
 */
struct GpuComplexStorage : public ComplexStorage, public GpuStorage {
  ComplexPtr array;

  GpuComplexStorage(tcapint n, int64_t did, bool alloc = true)
      : ComplexStorage(DeviceTag::GPU, n), array(nullptr, [](complex *) {}) {
    dev = OCLEngine::Instance().GetWeedDevice(did);
    if (alloc) {
      AddAlloc(sizeof(complex) * size);
      buffer = MakeBuffer(n);
    }
  }

  GpuComplexStorage(const std::vector<complex> &val, int64_t did)
      : ComplexStorage(DeviceTag::GPU, val.size()), array(Alloc(val.size())) {
    dev = OCLEngine::Instance().GetWeedDevice(did);
    AddAlloc(sizeof(complex) * size);
    std::copy(val.begin(), val.end(), array.get());
    buffer = MakeBuffer(val.size());
    if (!(dev->device_context->use_host_mem)) {
      array.reset();
    }
  }

  virtual ~GpuComplexStorage() {
    if (is_mapped) {
      dev->UnlockSync(buffer, array.get());
    }
    SubtractAlloc(sizeof(complex) * size);
  }

  void FillZeros() override { dev->ClearRealBuffer(buffer, size << 1U); }
  void FillOnes() override { dev->FillOnesComplex(buffer, size); }
  void FillValue(complex v) override { dev->FillValueComplex(buffer, size, v); }

  StoragePtr Upcast(DType dt) { return get_ptr(); };

  BufferPtr MakeBuffer(tcapint n) {
    if (dev->device_context->use_host_mem) {
      if (!array) {
        array = Alloc(n);
      }

      return dev->MakeBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             sizeof(complex) * n, array.get());
    }

    if (!array) {
      return dev->MakeBuffer(CL_MEM_READ_WRITE, sizeof(complex) * n);
    }

    return dev->MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(complex) * n, array.get());
  }

  complex operator[](tcapint idx) {
    if (idx >= size) {
      throw std::invalid_argument(
          "GpuComplexStorage::operator[] argument out-of-bounds!");
    }

    tcapint i = idx << 1U;

    return complex(dev->GetReal(buffer, i), dev->GetReal(buffer, i + 1U));
  }

  void write(tcapint idx, complex val) {
    throw std::domain_error("Don't use GPU-based ComplexStorage::write()!");
  }

  void add(tcapint idx, complex val) {
    throw std::domain_error("Don't use GPU-based ComplexStorage::add()!");
  }

  StoragePtr cpu() override;
  StoragePtr gpu(int64_t did = -1) override { return get_ptr(); };
};
typedef std::shared_ptr<GpuComplexStorage> GpuComplexStoragePtr;
} // namespace Weed
