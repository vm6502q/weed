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

#include "ops/gpu_flat.hpp"
#include "storage/gpu_complex_storage.hpp"
#include "storage/gpu_real_storage.hpp"

namespace Weed {
bool gpu_is_flat(const BaseTensor &t) {
  if (t.is_scalar()) {
    return true;
  }

  // Contiguous, in column-major order (An index of extent 1 doesn't matter.)
  tcapint st = 1U;
  for (size_t i = 0U; i < t.shape.size(); ++i) {
    if (t.shape[i] == 1U) {
      continue;
    }
    if (t.stride[i] != st) {
      return false;
    }
    st *= t.shape[i];
  }

  return true;
}

void gpu_require_flat(const BaseTensor &t, const char *op) {
  if (!gpu_is_flat(t)) {
    throw std::invalid_argument(
        std::string("In ") + op +
        ", the GPU output tensor must be contiguous (or a scalar)!");
  }
}

// OpenCL buffers for the shape and stride of a tensor
static void make_index_buffers(GpuDevicePtr dev, const BaseTensor &t,
                               BufferPtr &shapeBuffer,
                               BufferPtr &strideBuffer) {
  const cl_mem_flags flags = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
  shapeBuffer = dev->MakeBuffer(flags, sizeof(tcapint) * t.shape.size(),
                                (void *)&(t.shape[0U]));
  strideBuffer = dev->MakeBuffer(flags, sizeof(tcapint) * t.stride.size(),
                                 (void *)&(t.stride[0U]));
}

template <typename S>
static Tensor gather(const Tensor &t, const OCLAPI &api_call) {
  std::shared_ptr<S> src = std::dynamic_pointer_cast<S>(t.storage);
  if (!src) {
    throw std::invalid_argument("gpu_flat() storage type mismatch!");
  }

  const tcapint n = t.get_broadcast_size();
  std::shared_ptr<S> dst = std::make_shared<S>(n, src->get_device_id());

  BufferPtr shapeBuffer, strideBuffer;
  make_index_buffers(src->dev, t, shapeBuffer, strideBuffer);

  const tcapint args[12U]{
      t.offset, (tcapint)t.shape.size(), 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
      0U};
  src->dev->RequestKernel(
      api_call, args, n, {src->buffer, dst->buffer, shapeBuffer, strideBuffer});

  Tensor f;
  f.storage = dst;
  f.offset = 0U;
  f.shape = t.shape;
  f.stride = BaseTensor::full_contiguous_stride(t.shape);
  f.requires_grad = false;

  return f;
}

static Tensor gather_any(const Tensor &t) {
  if (t.storage->dtype == DType::COMPLEX) {
    return gather<GpuComplexStorage>(t, OCLAPI::OCL_API_GATHER_COMPLEX);
  }

  return gather<GpuRealStorage>(t, OCLAPI::OCL_API_GATHER_REAL);
}

Tensor gpu_contiguous(const Tensor &t) {
  if (t.is_scalar() && (t.get_broadcast_size() > 1U)) {
    return gather_any(t);
  }

  return gpu_flat(t);
}

Tensor gpu_flat(const Tensor &t) {
  if (gpu_is_flat(t)) {
    Tensor f;
    f.storage = t.storage;
    f.offset = t.offset;
    f.shape = t.shape;
    f.stride = t.stride;
    f.requires_grad = false;

    return f;
  }

  return gather_any(t);
}

template <typename S>
static void scatter(Tensor &dst, const Tensor &src, const OCLAPI &api_call) {
  std::shared_ptr<S> d = std::dynamic_pointer_cast<S>(dst.storage);
  std::shared_ptr<S> s = std::dynamic_pointer_cast<S>(src.storage);
  if (!d || !s) {
    throw std::invalid_argument("gpu_scatter() storage type mismatch!");
  }

  BufferPtr shapeBuffer, strideBuffer;
  make_index_buffers(d->dev, dst, shapeBuffer, strideBuffer);

  const tcapint args[12U]{dst.offset, (tcapint)dst.shape.size(),
                          src.offset, gpu_step(src),
                          0U,         0U,
                          0U,         0U,
                          0U,         0U,
                          0U,         0U};
  d->dev->RequestKernel(api_call, args, dst.get_broadcast_size(),
                        {d->buffer, s->buffer, shapeBuffer, strideBuffer});
}

void gpu_scatter(Tensor &dst, const Tensor &src) {
  if (dst.get_broadcast_size() != src.get_broadcast_size()) {
    throw std::invalid_argument("gpu_scatter() sizes do not match!");
  }
  if (!gpu_is_flat(src)) {
    throw std::invalid_argument("gpu_scatter() source must be flat!");
  }

  if (dst.storage->dtype == DType::COMPLEX) {
    scatter<GpuComplexStorage>(dst, src, OCLAPI::OCL_API_SCATTER_COMPLEX);
  } else {
    scatter<GpuRealStorage>(dst, src, OCLAPI::OCL_API_SCATTER_REAL);
  }
}

Tensor gpu_broadcast_view(const Tensor &t, const std::vector<tcapint> &shape) {
  if (t.shape.size() != shape.size()) {
    throw std::invalid_argument("gpu_broadcast_view() rank mismatch!");
  }

  Tensor v;
  v.storage = t.storage;
  v.offset = t.offset;
  v.shape = shape;
  v.stride = t.stride;
  v.requires_grad = false;
  for (size_t i = 0U; i < shape.size(); ++i) {
    if (t.shape[i] == shape[i]) {
      continue;
    }
    if (t.shape[i] != 1U) {
      throw std::invalid_argument("gpu_broadcast_view() shape mismatch!");
    }
    v.stride[i] = 0U;
  }

  return v;
}
} // namespace Weed
