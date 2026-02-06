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

#include "devices/gpu_device.hpp"

#define DISPATCH_TEMP_WRITE(waitVec, buff, size, array, clEvent)               \
  tryOcl("Failed to write buffer", [&] {                                       \
    return queue.enqueueWriteBuffer(buff, CL_FALSE, 0U, size, array,           \
                                    waitVec.get(), &clEvent);                  \
  });

#define DISPATCH_BLOCK_READ(waitVec, buff, offset, length, array)              \
  tryOcl("Failed to read buffer", [&] {                                        \
    return queue.enqueueReadBuffer(buff, CL_TRUE, offset, length, array,       \
                                   waitVec.get());                             \
  });                                                                          \
  wait_refs.clear();

namespace Weed {
void CL_CALLBACK _PopQueue(cl_event event, cl_int type, void *user_data) {
  ((GpuDevice *)user_data)->PopQueue(true);
}

BufferPtr GpuDevice::MakeBuffer(const cl_mem_flags &flags, const size_t &size,
                                void *host_ptr) {
  CheckCallbackError();

  cl_int error;
  BufferPtr toRet =
      std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
  if (error == CL_SUCCESS) {
    // Success
    return toRet;
  }

  // Soft finish (just for this GpuDevice)
  clFinish();

  toRet = std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
  if (error == CL_SUCCESS) {
    // Success after clearing GpuDevice queue
    return toRet;
  }

  // Hard finish (for the unique OpenCL device)
  clFinish(true);

  toRet = std::make_shared<cl::Buffer>(context, flags, size, host_ptr, &error);
  if (error != CL_SUCCESS) {
    if (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
      throw bad_alloc(
          "CL_MEM_OBJECT_ALLOCATION_FAILURE in GpuDevice::MakeBuffer()");
    }
    if (error == CL_OUT_OF_HOST_MEMORY) {
      throw bad_alloc("CL_OUT_OF_HOST_MEMORY in GpuDevice::MakeBuffer()");
    }
    if (error == CL_INVALID_BUFFER_SIZE) {
      throw bad_alloc("CL_INVALID_BUFFER_SIZE in GpuDevice::MakeBuffer()");
    }
    throw std::runtime_error(
        "OpenCL error code on buffer allocation attempt: " +
        std::to_string(error));
  }

  return toRet;
}

void GpuDevice::PopQueue(const bool &isDispatch) {
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

void GpuDevice::clFinish(const bool &doHard) {
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
void GpuDevice::tryOcl(const std::string &message,
                       const std::function<int()> &oclCall) {
  CheckCallbackError();

  if (oclCall() == CL_SUCCESS) {
    // Success
    return;
  }

  // Soft finish (just for this GpuDevice)
  clFinish();

  if (oclCall() == CL_SUCCESS) {
    // Success after clearing GpuDevice queue
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

void GpuDevice::DispatchQueue() {
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
  // same kernel therefore can't be used by other GpuDevice instances, until
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
  auto wic = !item.workItemCount2
                 ? cl::NDRange(item.workItemCount)
                 : cl::NDRange(item.workItemCount, item.workItemCount2);
  auto lgs = !item.localGroupSize2
                 ? cl::NDRange(item.localGroupSize)
                 : cl::NDRange(item.localGroupSize, item.localGroupSize2);
  EventVecPtr kernelWaitVec = ResetWaitEvents(false);
  cl_int error = CL_SUCCESS;
  device_context->EmplaceEvent([&](cl::Event &event) {
    event.setCallback(CL_COMPLETE, _PopQueue, this);
    error = queue.enqueueNDRangeKernel(
        ocl.call, cl::NullRange, // kernel, offset
        wic,                     // global number of work items
        lgs,                     // local number (per group)
        kernelWaitVec.get(),     // vector of events to wait for
        &event);                 // handle to wait for the kernel
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

EventVecPtr GpuDevice::ResetWaitEvents(const bool &waitQueue) {
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

PoolItemPtr GpuDevice::GetFreePoolItem() {
  std::lock_guard<std::mutex> lock(queue_mutex);

  CheckCallbackError();

  while (wait_queue_items.size() >= poolItems.size()) {
    poolItems.push_back(std::make_shared<PoolItem>(context));
  }

  return poolItems[wait_queue_items.size()];
}

inline size_t pick_group_size(const size_t &nwi) {
  if (nwi <= 32U) {
    return nwi;
  }

  size_t ngs = 32U;
  while (((nwi / ngs) * ngs) != nwi) {
    --ngs;
  }

  return ngs;
}

void GpuDevice::RequestKernel(const OCLAPI &api_call, const tcapint *vciArgs,
                              const size_t &nwi, std::vector<BufferPtr> buffers,
                              const size_t &nwi2, const complex *c) {
  EventVecPtr waitVec = ResetWaitEvents();
  PoolItemPtr poolItem = GetFreePoolItem();
  cl::Event writeArgsEvent;
  DISPATCH_TEMP_WRITE(waitVec, *(poolItem->vciBuffer),
                      sizeof(tcapint) * VCI_ARG_LEN, vciArgs, writeArgsEvent);
  cl::Event writeArgsEvent2;
  if (c) {
    DISPATCH_TEMP_WRITE(waitVec, *(poolItem->complexBuffer),
                        sizeof(complex) * CMPLX_ARG_LEN, c, writeArgsEvent2);
  }
  const size_t ngs = pick_group_size(nwi);
  const size_t ngs2 = (!nwi2) ? 0U : pick_group_size(nwi2);
  writeArgsEvent.wait();
  buffers.push_back(poolItem->vciBuffer);
  if (c) {
    writeArgsEvent2.wait();
    buffers.push_back(poolItem->complexBuffer);
  }
  QueueCall(api_call, nwi, ngs, buffers, nwi2, ngs2);
}

void GpuDevice::ClearIntBuffer(BufferPtr buffer, const size_t &nwi) {
  const size_t ngs = pick_group_size(nwi);
  QueueCall(OCLAPI::OCL_API_CLEAR_BUFFER_INT, nwi, ngs,
            std::vector<BufferPtr>{buffer});
}
void GpuDevice::ClearRealBuffer(BufferPtr buffer, const size_t &nwi) {
  const size_t ngs = pick_group_size(nwi);
  QueueCall(OCLAPI::OCL_API_CLEAR_BUFFER_REAL, nwi, ngs,
            std::vector<BufferPtr>{buffer});
}
void GpuDevice::FillOnesInt(BufferPtr buffer, const size_t &nwi) {
  const size_t ngs = pick_group_size(nwi);
  QueueCall(OCLAPI::OCL_API_FILL_ONES_INT, nwi, ngs,
            std::vector<BufferPtr>{buffer});
}
void GpuDevice::FillOnesReal(BufferPtr buffer, const size_t &nwi) {
  const size_t ngs = pick_group_size(nwi);
  QueueCall(OCLAPI::OCL_API_FILL_ONES_REAL, nwi, ngs,
            std::vector<BufferPtr>{buffer});
}
void GpuDevice::FillOnesComplex(BufferPtr buffer, const size_t &nwi) {
  const size_t ngs = pick_group_size(nwi);
  QueueCall(OCLAPI::OCL_API_FILL_ONES_COMPLEX, nwi, ngs,
            std::vector<BufferPtr>{buffer});
}
void GpuDevice::FillValueInt(BufferPtr buffer, const size_t &nwi,
                             const symint &v) {
  tcapint vciArgs[VCI_ARG_LEN] = {(tcapint)v, 0U, 0U, 0U, 0U,
                                  0U,         0U, 0U, 0U, 0U};
  EventVecPtr waitVec = ResetWaitEvents();
  PoolItemPtr poolItem = GetFreePoolItem();
  cl::Event writeArgsEvent;
  DISPATCH_TEMP_WRITE(waitVec, *(poolItem->vciBuffer), sizeof(tcapint), vciArgs,
                      writeArgsEvent);
  const size_t ngs = pick_group_size(nwi);
  writeArgsEvent.wait();
  QueueCall(OCLAPI::OCL_API_FILL_VALUE_INT, nwi, ngs,
            std::vector<BufferPtr>{buffer, poolItem->vciBuffer});
}
void GpuDevice::FillValueReal(BufferPtr buffer, const size_t &nwi,
                              const real1 &v) {
  complex cmplxArgs[CMPLX_ARG_LEN] = {(complex)v};
  EventVecPtr waitVec = ResetWaitEvents();
  PoolItemPtr poolItem = GetFreePoolItem();
  cl::Event writeArgsEvent;
  DISPATCH_TEMP_WRITE(waitVec, *(poolItem->complexBuffer),
                      sizeof(complex) * CMPLX_ARG_LEN, cmplxArgs,
                      writeArgsEvent);
  const size_t ngs = pick_group_size(nwi);
  writeArgsEvent.wait();
  QueueCall(OCLAPI::OCL_API_FILL_VALUE_REAL, nwi, ngs,
            std::vector<BufferPtr>{buffer, poolItem->complexBuffer});
}
void GpuDevice::FillValueComplex(BufferPtr buffer, const size_t &nwi,
                                 const complex &v) {
  complex cmplxArgs[CMPLX_ARG_LEN] = {v};
  EventVecPtr waitVec = ResetWaitEvents();
  PoolItemPtr poolItem = GetFreePoolItem();
  cl::Event writeArgsEvent;
  DISPATCH_TEMP_WRITE(waitVec, *(poolItem->complexBuffer),
                      sizeof(complex) * CMPLX_ARG_LEN, cmplxArgs,
                      writeArgsEvent);
  writeArgsEvent.wait();
  const size_t ngs = pick_group_size(nwi);
  QueueCall(OCLAPI::OCL_API_FILL_VALUE_COMPLEX, nwi, ngs,
            std::vector<BufferPtr>{buffer, poolItem->complexBuffer});
}
void GpuDevice::UpcastRealBuffer(BufferPtr buffer_in, BufferPtr buffer_out,
                                 const size_t &nwi) {
  const size_t ngs = pick_group_size(nwi);
  QueueCall(OCLAPI::OCL_API_REAL_TO_COMPLEX_BUFFER, nwi, ngs,
            std::vector<BufferPtr>{buffer_in, buffer_out});
}

real1 GpuDevice::GetInt(BufferPtr buffer, const tcapint &idx) {
  symint v;
  EventVecPtr waitVec = ResetWaitEvents();
  DISPATCH_BLOCK_READ(waitVec, *buffer, sizeof(symint) * idx, sizeof(symint),
                      &v);

  return v;
}
real1 GpuDevice::GetReal(BufferPtr buffer, const tcapint &idx) {
  real1 v;
  EventVecPtr waitVec = ResetWaitEvents();
  DISPATCH_BLOCK_READ(waitVec, *buffer, sizeof(real1) * idx, sizeof(real1), &v);

  return v;
}
complex GpuDevice::GetComplex(BufferPtr buffer, const tcapint &idx) {
  complex v;
  EventVecPtr waitVec = ResetWaitEvents();
  DISPATCH_BLOCK_READ(waitVec, *buffer, sizeof(complex) * idx, sizeof(complex),
                      &v);

  return v;
}
void GpuDevice::SetReal(const real1 &val, BufferPtr buffer,
                        const tcapint &idx) {
  EventVecPtr waitVec = ResetWaitEvents();
  device_context->EmplaceEvent(
      [this, val, buffer, idx, waitVec](cl::Event &event) {
        tryOcl("Failed to enqueue buffer write", [&] {
          return queue.enqueueWriteBuffer(*buffer, CL_FALSE,
                                          sizeof(real1) * idx, sizeof(real1),
                                          &val, waitVec.get(), &event);
        });
      });
}
void GpuDevice::SetComplex(const complex &val, BufferPtr buffer,
                           const tcapint &idx) {
  EventVecPtr waitVec = ResetWaitEvents();
  device_context->EmplaceEvent([this, val, buffer, idx,
                                waitVec](cl::Event &event) {
    tryOcl("Failed to enqueue buffer write", [&] {
      return queue.enqueueWriteBuffer(*buffer, CL_FALSE, sizeof(complex) * idx,
                                      sizeof(complex), &val, waitVec.get(),
                                      &event);
    });
  });
}
} // namespace Weed
