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
#include "storage/gpu_storage.hpp"
#include "storage/real_storage.hpp"

#if !ENABLE_OPENCL && !ENABLE_CUDA
#error GPU files were included without either OpenCL and CUDA enabled.
#endif

namespace Weed {
/**
 * GPU-accessible storage for real data type elements
 */
struct GpuRealStorage : public RealStorage, public GpuStorage {
  RealPtr array;

  GpuRealStorage(vecCapIntGpu n, int64_t did, bool alloc = true)
      : RealStorage(DeviceTag::GPU, n), array(nullptr, [](real1 *) {}) {
    dev = OCLEngine::Instance().GetWeedDevice(did);
    if (alloc) {
      AddAlloc(sizeof(real1) * size);
      buffer = MakeBuffer(n);
    }
  }

  GpuRealStorage(std::vector<real1> val, int64_t did)
      : RealStorage(DeviceTag::GPU, val.size()), array(Alloc(val.size())) {
    dev = OCLEngine::Instance().GetWeedDevice(did);
    AddAlloc(sizeof(real1) * size);
    std::copy(val.begin(), val.end(), array.get());
    buffer = MakeBuffer(val.size());
    array.reset();
  }

  virtual ~GpuRealStorage() {
    if (is_mapped) {
      dev->UnlockSync(buffer, array.get());
    }
    SubtractAlloc(sizeof(real1) * size);
  }

  void AddAlloc(size_t sz) {
    size_t currentAlloc =
        OCLEngine::Instance().AddToActiveAllocSize(dev->deviceID, sz);
    if (currentAlloc > dev->device_context->GetGlobalAllocLimit()) {
      OCLEngine::Instance().SubtractFromActiveAllocSize(dev->deviceID, sz);
      throw bad_alloc("VRAM limits exceeded in GpuComplexStorage::AddAlloc()");
    }
  }
  void SubtractAlloc(size_t sz) {
    OCLEngine::Instance().SubtractFromActiveAllocSize(dev->deviceID, sz);
  }

  int64_t get_device_id() override { return dev->deviceID; }

  void FillZeros() override { dev->ClearRealBuffer(buffer, size); }
  void FillOnes() override { dev->FillOnesReal(buffer, size); }
  void FillValue(real1 v) override { dev->FillValueReal(buffer, size, v); }

  StoragePtr Upcast(DType dt) override {
    if (dt == DType::REAL) {
      return get_ptr();
    }

    GpuComplexStoragePtr n =
        std::make_shared<GpuComplexStorage>(size, dev->deviceID);
    dev->UpcastRealBuffer(buffer, n->buffer, size);

    return n;
  };

  BufferPtr MakeBuffer(vecCapIntGpu n) {
    if (dev->device_context->use_host_mem) {
      if (!array) {
        array = Alloc(n);
      }

      return dev->MakeBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             sizeof(real1) * n, array.get());
    }

    if (!array) {
      return dev->MakeBuffer(CL_MEM_READ_WRITE, sizeof(real1) * n);
    }

    return dev->MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(real1) * n, array.get());
  }

  real1 operator[](vecCapInt idx) {
    if (idx >= size) {
      throw std::invalid_argument(
          "GpuRealStorage::operator[] argument out-of-bounds!");
    }

    return dev->GetReal(buffer, (vecCapIntGpu)idx);
  }

  StoragePtr cpu() override;
  StoragePtr gpu(int64_t did = -1) override { return get_ptr(); };
};
typedef std::shared_ptr<GpuRealStorage> GpuRealStoragePtr;
} // namespace Weed
