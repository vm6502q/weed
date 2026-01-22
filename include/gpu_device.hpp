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

#define DISPATCH_TEMP_WRITE(waitVec, buff, size, array, clEvent)               \
  tryOcl("Failed to write buffer", [&] {                                       \
    return queue.enqueueWriteBuffer(buff, CL_FALSE, 0U, size, array,           \
                                    waitVec.get(), &clEvent);                  \
  });

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
      : callbackError(CL_SUCCESS), totalOclAllocSize(0U), deviceID(did) {
    const size_t deviceCount = OCLEngine::Instance().GetDeviceCount();

    if (!deviceCount) {
      throw std::runtime_error("GpuDevice::GpuDevice(): No available devices.");
    }

    if (did > ((int64_t)deviceCount)) {
      throw std::runtime_error(
          "GpuDevice::GpuDevice(): Requested device doesn't exist.");
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

  void AddQueueItem(const QueueItem &item) {
    bool isBase;
    // For lock_guard:
    if (true) {
      std::lock_guard<std::mutex> lock(queue_mutex);
      CheckCallbackError();
      isBase = wait_queue_items.empty();
      wait_queue_items.push_back(item);
    }

    if (isBase) {
      DispatchQueue();
    }
  }

  void QueueCall(OCLAPI api_call, size_t workItemCount, size_t localGroupSize,
                 std::vector<BufferPtr> args, size_t localBuffSize = 0U,
                 size_t deallocSize = 0U) {
    if (localBuffSize > device_context->GetLocalSize()) {
      throw bad_alloc(
          "Local memory limits exceeded in QEngineOCL::QueueCall()");
    }
    AddQueueItem(QueueItem(api_call, workItemCount, localGroupSize, deallocSize,
                           args, localBuffSize));
  }

  PoolItemPtr GetFreePoolItem();

  void RequestKernel(OCLAPI api_call, const vecCapIntGpu *bciArgs,
                     const size_t nwi, std::vector<BufferPtr> buffers);

  void ClearRealBuffer(BufferPtr buffer, const size_t nwi);
  void FillOnesReal(BufferPtr buffer, const size_t nwi);
  void FillOnesComplex(BufferPtr buffer, const size_t nwi);
  void UpcastRealBuffer(BufferPtr buffer_in, BufferPtr buffer_out,
                        const size_t nwi);
};
} // namespace Weed
