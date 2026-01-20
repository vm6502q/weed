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

#include "pool_item.hpp"
#include "queue_item.hpp"

#if !ENABLE_OPENCL && !ENABLE_CUDA
#error GPU files were included without either OpenCL and CUDA enabled.
#endif

#include <list>

namespace Weed {
struct GpuDevice {
  cl_int callbackError;
  size_t totalOclAllocSize;
  int64_t deviceID;
  DeviceContextPtr device_context;
  cl::Context context;
  cl::CommandQueue queue;
  std::mutex queue_mutex;
  std::vector<EventVecPtr> wait_refs;
  std::list<QueueItem> wait_queue_items;
  std::vector<PoolItemPtr> poolItems;

  GpuDevice(int64_t did = -1)
    : callbackError(CL_SUCCESS)
    , totalOclAllocSize(0U)
    , deviceID(did)
  {
    const size_t deviceCount = OCLEngine::Instance().GetDeviceCount();

    if (!deviceCount) {
        throw std::runtime_error("GpuDevice::GpuDevice(): No available devices.");
    }

    if (did > ((int64_t)deviceCount)) {
        throw std::runtime_error("GpuDevice::GpuDevice(): Requested device doesn't exist.");
    }

    clFinish();

    device_context = OCLEngine::Instance().GetDeviceContextPtr(did);
    context = device_context->context;
    queue = device_context->queue;
  }
  ~GpuDevice() {}

  BufferPtr MakeBuffer(cl_mem_flags flags, size_t size,
                       void *host_ptr = nullptr);

  void clFinish(bool doHard = false);
  void tryOcl(std::string message, std::function<int()> oclCall);
  void PopQueue(bool isDispatch);
  void DispatchQueue();
  EventVecPtr ResetWaitEvents(bool waitQueue = true);

  void CheckCallbackError() {
    if (callbackError == CL_SUCCESS) {
      return;
    }

    wait_queue_items.clear();
    wait_refs.clear();

    throw std::runtime_error("Failed to enqueue kernel, error code: " +
                             std::to_string(callbackError));
  }

  void AddAlloc(size_t size) {
    size_t currentAlloc =
        OCLEngine::Instance().AddToActiveAllocSize(deviceID, size);
    if (device_context &&
        (currentAlloc > device_context->GetGlobalAllocLimit())) {
      OCLEngine::Instance().SubtractFromActiveAllocSize(deviceID, size);
      throw bad_alloc("VRAM limits exceeded in QEngineOCL::AddAlloc()");
    }
    totalOclAllocSize += size;
  }
  void SubtractAlloc(size_t size) {
    OCLEngine::Instance().SubtractFromActiveAllocSize(deviceID, size);
    totalOclAllocSize -= size;
  }
};
typedef std::shared_ptr<GpuDevice> GpuDevicePtr;
} // namespace Weed
