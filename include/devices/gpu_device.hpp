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
/**
 * Manages a GPU (or accelerator) device context, per discrete (or integrated)
 * device, or per device queue of dependent events
 */
struct GpuDevice {
  /**
   * Unique hardware device identifier
   */
  int64_t deviceID;

  cl_int callbackError;
  size_t totalOclAllocSize;
  DeviceContextPtr device_context;
  cl::Context context;
  cl::CommandQueue queue;
  std::mutex queue_mutex;
  std::vector<EventVecPtr> wait_refs;
  std::list<QueueItem> wait_queue_items;
  std::vector<PoolItemPtr> poolItems;

  /**
   * Create a context to manage a specific device ID or device queue of
   * dependent events
   */
  GpuDevice(const int64_t &did = -1)
      : deviceID(did), callbackError(CL_SUCCESS), totalOclAllocSize(0U) {
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

  /**
   * Create a buffer on this device context
   */
  BufferPtr MakeBuffer(const cl_mem_flags &flags, const size_t &size,
                       void *host_ptr = nullptr);

  /**
   * Flush and finish queue of dependent events (or flush and finish for entire
   * hardware device with doHard=True)
   */
  void clFinish(const bool &doHard = false);
  /**
   * Error-handling wrapper for OpenCL calls
   */
  void tryOcl(const std::string &message, const std::function<int()> &oclCall);
  /**
   * For kernel callback to free resources and start the next queue event
   */
  void PopQueue(const bool &isDispatch);
  /**
   * Start kernel dispatch and callback cycle
   */
  void DispatchQueue();
  /**
   * Get dependent events to wait for next and clear the wait events buffer
   */
  EventVecPtr ResetWaitEvents(const bool &waitQueue = true);

  /**
   * Handle any OpenCL error on kernel callback
   */
  void CheckCallbackError() {
    if (callbackError == CL_SUCCESS) {
      return;
    }

    wait_queue_items.clear();
    wait_refs.clear();

    throw std::runtime_error("Failed to enqueue kernel, error code: " +
                             std::to_string(callbackError));
  }

  /**
   * Add byte count to manual GPU memory tracking
   */
  void AddAlloc(const size_t &size) {
    size_t currentAlloc =
        OCLEngine::Instance().AddToActiveAllocSize(deviceID, size);
    if (device_context &&
        (currentAlloc > device_context->GetGlobalAllocLimit())) {
      OCLEngine::Instance().SubtractFromActiveAllocSize(deviceID, size);
      throw bad_alloc("VRAM limits exceeded in QEngineOCL::AddAlloc()");
    }
    totalOclAllocSize += size;
  }

  /**
   * Subtract byte count to manual GPU memory tracking
   */
  void SubtractAlloc(const size_t &size) {
    OCLEngine::Instance().SubtractFromActiveAllocSize(deviceID, size);
    totalOclAllocSize -= size;
  }

  /**
   * Insert a new item into the kernel callback queue
   */
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

  /**
   * Map or copy a GPU buffer into (CPU-accessible) host memory
   */
  bool LockSync(BufferPtr buffer, const size_t &sz, void *array,
                bool allow_lock = true) {
    EventVecPtr waitVec = ResetWaitEvents();

    if (allow_lock && device_context->use_host_mem) {
      // Needs unlock
      tryOcl("Failed to map buffer", [&] {
        cl_int error;
        queue.enqueueMapBuffer(*buffer, CL_TRUE, CL_MAP_READ, 0U, sz,
                               waitVec.get(), nullptr, &error);
        return error;
      });
      wait_refs.clear();
    } else {
      // Just a copy: don't need to unlock
      tryOcl("Failed to read buffer", [&] {
        return queue.enqueueReadBuffer(*buffer, CL_TRUE, 0U, sz, array,
                                       waitVec.get());
      });
      wait_refs.clear();
    }

    return device_context->use_host_mem;
  }

  /**
   * If LockSync() mapped a buffer rather than copying, unmap it for GPU use
   */
  void UnlockSync(BufferPtr buffer, void *array) {
    EventVecPtr waitVec = ResetWaitEvents();
    cl::Event unmapEvent;
    tryOcl("Failed to unmap buffer", [&] {
      return queue.enqueueUnmapMemObject(*buffer, array, waitVec.get(),
                                         &unmapEvent);
    });
    unmapEvent.wait();
    wait_refs.clear();
  }

  /**
   * Construct a new queue item and insert it into the kernel callback queue
   */
  void QueueCall(const OCLAPI &api_call, const size_t &workItemCount,
                 const size_t &localGroupSize,
                 const std::vector<BufferPtr> &args, const size_t &wic2 = 0U,
                 const size_t &lgs2 = 0U, const size_t &localBuffSize = 0U,
                 const size_t &deallocSize = 0U) {
    if (localBuffSize > device_context->GetLocalSize()) {
      throw bad_alloc(
          "Local memory limits exceeded in QEngineOCL::QueueCall()");
    }
    AddQueueItem(QueueItem(api_call, workItemCount, localGroupSize, deallocSize,
                           args, wic2, lgs2, localBuffSize));
  }

  /**
   * Reuse or construct a new queue item holder that isn't in use
   */
  PoolItemPtr GetFreePoolItem();

  /**
   * Construct a new queue item for a kernel call and insert it into the kernel
   * callback queue
   */
  void RequestKernel(const OCLAPI &api_call, const tcapint *vciArgs,
                     const size_t &nwi, std::vector<BufferPtr> buffers,
                     const size_t &nwi2 = 0U, const complex *c = nullptr);

  /**
   * Request a buffer zeroing in the queue (OpenCL v1.1 compatible style)
   */
  void ClearIntBuffer(BufferPtr buffer, const size_t &nwi);
  /**
   * Request a buffer zeroing in the queue (OpenCL v1.1 compatible style)
   */
  void ClearRealBuffer(BufferPtr buffer, const size_t &nwi);
  /**
   * Fill a buffer of Weed:symint elements with 1
   */
  void FillOnesInt(BufferPtr buffer, const size_t &nwi);
  /**
   * Fill a buffer of Weed:real1 elements with 1.0
   */
  void FillOnesReal(BufferPtr buffer, const size_t &nwi);
  /**
   * Fill a buffer of Weed:complex elements with 1.0
   */
  void FillOnesComplex(BufferPtr buffer, const size_t &nwi);
  /**
   * Fill a buffer of Weed:real1 elements with specified value
   */
  void FillValueInt(BufferPtr buffer, const size_t &nwi, const symint &v);
  /**
   * Fill a buffer of Weed:real1 elements with specified value
   */
  void FillValueReal(BufferPtr buffer, const size_t &nwi, const real1 &v);
  /**
   * Fill a buffer of Weed:complex elements with specified value
   */
  void FillValueComplex(BufferPtr buffer, const size_t &nwi, const complex &v);
  /**
   * Up-cast a real1 buffer to a (double-stride) complex buffer with the same
   * values
   */
  void UpcastRealBuffer(BufferPtr buffer_in, BufferPtr buffer_out,
                        const size_t &nwi);

  /**
   * Read a single real1 from a buffer
   */
  real1 GetInt(BufferPtr buffer, const tcapint &idx);
  /**
   * Read a single real1 from a buffer
   */
  real1 GetReal(BufferPtr buffer, const tcapint &idx);
  /**
   * Read a single complex from a buffer
   */
  complex GetComplex(BufferPtr buffer, const tcapint &idx);
  /**
   * Write a single real1 to a buffer
   */
  void SetInt(const real1 &val, BufferPtr buffer, const tcapint &idx);
  /**
   * Write a single real1 to a buffer
   */
  void SetReal(const real1 &val, BufferPtr buffer, const tcapint &idx);
  /**
   * Write a single complex to a buffer
   */
  void SetComplex(const complex &val, BufferPtr buffer, const tcapint &idx);
};
} // namespace Weed
