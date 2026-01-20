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

#include "common/oclengine.hpp"
#include "storage.hpp"

namespace Weed {
typedef std::unique_ptr<cl::Buffer> BufferPtr;

struct GpuStorage : Storage {
  BufferPtr buffer;
  cl_int callbackError;
  cl::CommandQueue queue;
  cl::Context context;
  DeviceContextPtr device_context;
  std::vector<EventVecPtr> wait_refs;
  std::list<QueueItem> wait_queue_items;

  GpuStorage() : buffer(nullptr) { device = DeviceTag::GPU; }

  ~GpuStorage() {}

  void CheckCallbackError() {
    if (callbackError == CL_SUCCESS) {
      return;
    }

    wait_queue_items.clear();
    wait_refs.clear();

    throw std::runtime_error("Failed to enqueue kernel, error code: " +
                             std::to_string(callbackError));
  }

  void clFinish(bool doHard) {
    if (!device_context) {
      return;
    }

    CheckCallbackError();

    while (wait_queue_items.size() > 1) {
      device_context->WaitOnAllEvents();
      PopQueue(true);
      CheckCallbackError();
    }

    if (doHard) {
      tryOcl("Failed to finish queue", [&] { return queue.finish(); });
    } else {
      device_context->WaitOnAllEvents();
      CheckCallbackError();
    }

    wait_refs.clear();
  }

  BufferPtr MakeBuffer(cl_mem_flags flags, size_t size,
                       void *host_ptr = nullptr) {
    CheckCallbackError();

    cl_int error;
    BufferPtr toRet =
        std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
    if (error == CL_SUCCESS) {
      // Success
      return toRet;
    }

    // Soft finish (just for this GpuStorage)
    clFinish();

    toRet =
        std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
    if (error == CL_SUCCESS) {
      // Success after clearing GpuStorage queue
      return toRet;
    }

    // Hard finish (for the unique OpenCL device)
    clFinish(true);

    toRet =
        std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
    if (error != CL_SUCCESS) {
      if (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
        throw bad_alloc(
            "CL_MEM_OBJECT_ALLOCATION_FAILURE in GpuStorage::MakeBuffer()");
      }
      if (error == CL_OUT_OF_HOST_MEMORY) {
        throw bad_alloc("CL_OUT_OF_HOST_MEMORY in GpuStorage::MakeBuffer()");
      }
      if (error == CL_INVALID_BUFFER_SIZE) {
        throw bad_alloc("CL_INVALID_BUFFER_SIZE in GpuStorage::MakeBuffer()");
      }
      throw std::runtime_error(
          "OpenCL error code on buffer allocation attempt: " +
          std::to_string(error));
    }

    return toRet;
  }
};
} // namespace Weed
