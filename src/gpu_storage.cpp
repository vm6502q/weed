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

#include "gpu_storage.hpp"

namespace Weed {
void CL_CALLBACK _PopQueue(cl_event event, cl_int type, void *user_data) {
  ((GpuStorage *)user_data)->PopQueue(true);
}

BufferPtr GpuStorage::MakeBuffer(cl_mem_flags flags, size_t size,
                                 void *host_ptr) {
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

  toRet = std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
  if (error == CL_SUCCESS) {
    // Success after clearing GpuStorage queue
    return toRet;
  }

  // Hard finish (for the unique OpenCL device)
  clFinish(true);

  toRet = std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
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

void GpuStorage::PopQueue(bool isDispatch) {
  // For lock_guard scope
  if (true) {
    std::lock_guard<std::mutex> lock(queue_mutex);

    if (poolItems.size()) {
      if (poolItems.size() > 1) {
        rotate(poolItems.begin(), poolItems.begin() + 1, poolItems.end());
      }
    }

    if (wait_queue_items.empty()) {
      return;
    }

    SubtractAlloc(wait_queue_items.front().deallocSize);
    wait_queue_items.pop_front();
  }

  if (callbackError != CL_SUCCESS) {
    wait_queue_items.clear();
    wait_refs.clear();

    return;
  }

  if (isDispatch) {
    DispatchQueue();
  }
}

void GpuStorage::clFinish(bool doHard) {
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

// For std::function, cl_int use might discard int qualifiers.
void GpuStorage::tryOcl(std::string message, std::function<int()> oclCall) {
  CheckCallbackError();

  if (oclCall() == CL_SUCCESS) {
    // Success
    return;
  }

  // Soft finish (just for this GpuStorage)
  clFinish();

  if (oclCall() == CL_SUCCESS) {
    // Success after clearing GpuStorage queue
    return;
  }

  // Hard finish (for the unique OpenCL device)
  clFinish(true);

  cl_int error = oclCall();
  if (error == CL_SUCCESS) {
    // Success after clearing all queues for the OpenCL device
    return;
  }

  wait_queue_items.clear();
  wait_refs.clear();

  // We're fatally blocked. Throw to exit.
  throw std::runtime_error(message + ", error code: " + std::to_string(error));
}

void GpuStorage::DispatchQueue() {
  QueueItem item;

  if (true) {
    std::lock_guard<std::mutex> lock(queue_mutex);

    if (wait_queue_items.empty()) {
      return;
    }

    item = wait_queue_items.front();
  }

  std::vector<BufferPtr> args = item.buffers;

  // We have to reserve the kernel, because its argument hooks are unique. The
  // same kernel therefore can't be used by other GpuStorage instances, until
  // we're done queueing it.
  OCLDeviceCall ocl = device_context->Reserve(item.api_call);

  // Load the arguments.
  for (unsigned int i = 0U; i < args.size(); ++i) {
    ocl.call.setArg(i, *args[i]);
  }

  // For all of our kernels, if a local memory buffer is used, there is always
  // only one, as the last argument.
  if (item.localBuffSize) {
    ocl.call.setArg(args.size(), cl::Local(item.localBuffSize));
  }

  // Dispatch the primary kernel, to apply the gate.
  EventVecPtr kernelWaitVec = ResetWaitEvents(false);
  cl_int error = CL_SUCCESS;
  device_context->EmplaceEvent([&](cl::Event &event) {
    event.setCallback(CL_COMPLETE, _PopQueue, this);
    error = queue.enqueueNDRangeKernel(
        ocl.call, cl::NullRange,          // kernel, offset
        cl::NDRange(item.workItemCount),  // global number of work items
        cl::NDRange(item.localGroupSize), // local number (per group)
        kernelWaitVec.get(),              // vector of events to wait for
        &event);                          // handle to wait for the kernel
  });
  if (error != CL_SUCCESS) {
    // We're fatally blocked, since we can't make any blocking calls like
    // clFinish() in a callback.
    callbackError = error;
    wait_queue_items.clear();
    wait_refs.clear();

    return;
  }
  error = queue.flush();
  if (error != CL_SUCCESS) {
    // We're fatally blocked, since we can't make any blocking calls like
    // clFinish() in a callback.
    callbackError = error;
    wait_queue_items.clear();
    wait_refs.clear();

    return;
  }
}

EventVecPtr GpuStorage::ResetWaitEvents(bool waitQueue) {
  if (waitQueue) {
    while (wait_queue_items.size() > 1) {
      device_context->WaitOnAllEvents();
      PopQueue(true);
      CheckCallbackError();
    }
  }
  EventVecPtr waitVec = device_context->ResetWaitEvents();
  if (waitVec->size()) {
    wait_refs.emplace_back(waitVec);
  }
  return wait_refs.empty() ? std::make_shared<EventVec>() : wait_refs.back();
}
} // namespace Weed
