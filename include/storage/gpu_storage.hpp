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
#include "storage/typed_storage.hpp"

#if !ENABLE_OPENCL && !ENABLE_CUDA
#error GPU files were included without either OpenCL and CUDA enabled.
#endif

namespace Weed {
/**
 * GPU-accessible storage
 */
template <typename T> struct GpuStorage : TypedStorage<T> {
  std::unique_ptr<T[], void (*)(T *)> data;
  GpuDevicePtr dev;
  BufferPtr buffer;

  GpuStorage(const StorageType &stp, const tcapint &n, const int64_t &did,
             const bool &alloc = true)
      : TypedStorage<T>(stp, DeviceTag::GPU, n), data(nullptr, [](T *) {}) {
    dev = OCLEngine::Instance().GetWeedDevice(did);
    AddAlloc(sizeof(T) * TypedStorage<T>::size);
    if (alloc) {
      buffer = MakeBuffer(n);
    }
  }

  GpuStorage(const StorageType &stp, const std::vector<T> &val,
             const int64_t &did)
      : TypedStorage<T>(stp, DeviceTag::GPU, val.size()),
        data(TypedStorage<T>::Alloc(val.size())) {
    dev = OCLEngine::Instance().GetWeedDevice(did);
    AddAlloc(sizeof(T) * TypedStorage<T>::size);
    std::copy(val.begin(), val.end(), data.get());
    buffer = MakeBuffer(val.size());
    if (!(dev->device_context->use_host_mem)) {
      data = nullptr;
    }
  }

  virtual ~GpuStorage() { SubtractAlloc(sizeof(T) * TypedStorage<T>::size); }

  int64_t get_device_id() const override { return dev->deviceID; }

  void AddAlloc(const size_t &sz) { dev->AddAlloc(sz); }
  void SubtractAlloc(const size_t &sz) { dev->SubtractAlloc(sz); }

  BufferPtr MakeBuffer(const tcapint &n) {
    if (dev->device_context->use_host_mem) {
      if (!data) {
        data = TypedStorage<T>::Alloc(n);
      }

      return dev->MakeBuffer(CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             sizeof(T) * n, data.get());
    }

    if (!data) {
      return dev->MakeBuffer(CL_MEM_READ_WRITE, sizeof(T) * n);
    }

    return dev->MakeBuffer(CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
                           sizeof(T) * n, data.get());
  }

  T operator[](const tcapint &idx) const = 0;

  void write(const tcapint &idx, const T &val) override {
    throw std::domain_error("Don't use GPU-based Storage::write()!");
  }

  void add(const tcapint &idx, const T &val) override {
    throw std::domain_error("Don't use GPU-based Storage::add()!");
  }

  StoragePtr gpu(const int64_t &did = -1) override {
    return TypedStorage<T>::get_ptr();
  }
};
} // namespace Weed
