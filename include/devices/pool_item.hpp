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

#include "common/weed_types.hpp"

#if !ENABLE_GPU
#error GPU files were included without either OpenCL and CUDA enabled.
#endif

#include "common/oclengine.hpp"

#include <algorithm>
#include <vector>

#define CMPLX_ARG_LEN 1
#define VCI_ARG_LEN 10

namespace Weed {
/**
 * OpenCL bad_alloc wrapper with informative error message
 */
struct bad_alloc : public std::bad_alloc {
  std::string m;

  bad_alloc(std::string message) : m(message) {
    // Intentionally left blank.
  }

  const char *what() const noexcept { return m.c_str(); }
};

/**
 * Queue pool item for OpenCL kernel callback cycle
 */
struct PoolItem {
  BufferPtr complexBuffer;
  BufferPtr vciBuffer;

#if ENABLE_OPENCL
  PoolItem(cl::Context &context) {
    complexBuffer = MakeBuffer(context, sizeof(complex) * CMPLX_ARG_LEN);
    vciBuffer = MakeBuffer(context, sizeof(tcapint) * VCI_ARG_LEN);
  }

  BufferPtr MakeBuffer(const cl::Context &context, size_t size) {
    cl_int error;
    BufferPtr toRet = std::unique_ptr<cl::Buffer>(new cl::Buffer(
        context, CL_MEM_READ_ONLY, size, (void *)nullptr, &error));
    if (error != CL_SUCCESS) {
      if (error == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
        throw bad_alloc(
            "CL_MEM_OBJECT_ALLOCATION_FAILURE in PoolItem::MakeBuffer()");
      }
      if (error == CL_OUT_OF_HOST_MEMORY) {
        throw bad_alloc("CL_OUT_OF_HOST_MEMORY in PoolItem::MakeBuffer()");
      }
      if (error == CL_INVALID_BUFFER_SIZE) {
        throw bad_alloc("CL_INVALID_BUFFER_SIZE in PoolItem::MakeBuffer()");
      }
      throw std::runtime_error(
          "OpenCL error code on buffer allocation attempt: " +
          std::to_string(error));
    }

    return toRet;
  }
#else
  PoolItem() {
    complexBuffer = MakeBuffer(sizeof(complex) * CMPLX_ARG_LEN);
    vciBuffer = MakeBuffer(sizeof(tcapint) * VCI_ARG_LEN);
  }

  BufferPtr MakeBuffer(size_t size) {
    cudaError_t error;

    BufferPtr toRet = std::shared_ptr<void>(AllocRaw(size, &error),
                                            [](void *c) { cudaFree(c); });

    if (error != cudaSuccess) {
      throw std::runtime_error(
          "CUDA error code on buffer allocation attempt: " +
          std::to_string(error));
    }

    return toRet;
  }

  void *AllocRaw(size_t size, cudaError_t *errorPtr) {
    void *toRet;
    *errorPtr = cudaMalloc(&toRet, size);

    return toRet;
  }
#endif
};

typedef std::shared_ptr<PoolItem> PoolItemPtr;
} // namespace Weed
