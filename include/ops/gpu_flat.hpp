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
#include "tensors/symbol_tensor.hpp"
#include "tensors/tensor.hpp"

#if !ENABLE_GPU
#error GPU files were included without either OpenCL and CUDA enabled.
#endif

namespace Weed {
/**
 * The element-wise GPU kernels address every operand as
 *
 *     buffer[offset + i * step]
 *
 * for the flat (column-major) element index "i". That can only express two
 * layouts: a contiguous tensor (step 1), or one element that is broadcast to
 * every index (step 0).
 *
 * Note that stride[0] is NOT that step: an index of extent 1 has stride 0, so
 * a contiguous tensor of shape (1, T, D) has stride[0] == 0, but step 1.
 */
inline tcapint gpu_step(const BaseTensor &t) { return t.is_scalar() ? 0U : 1U; }

/**
 * Can the element-wise GPU kernels address this tensor directly? (A partial
 * broadcast, like a bias over (B, T, D), or a transposed or strided view,
 * cannot be addressed with a single step.)
 */
bool gpu_is_flat(const BaseTensor &t);

/**
 * Throw unless the element-wise GPU kernels can address this tensor directly.
 */
void gpu_require_flat(const BaseTensor &t, const char *op);

/**
 * Return a tensor with the same elements as "t" that the element-wise GPU
 * kernels can address. This is "t" itself if it is already flat; otherwise it
 * is a contiguous copy, gathered on the device.
 */
Tensor gpu_flat(const Tensor &t);

/**
 * Like gpu_flat(), but a scalar that is broadcast to more than one element is
 * also copied out, so that the result is always contiguous (with step 1). The
 * kernels that work along one index of a tensor (reductions, softmax,
 * embedding) need this.
 */
Tensor gpu_contiguous(const Tensor &t);

/**
 * Product of the extents before index "axis": the storage step along "axis"
 * of a contiguous (column-major) tensor
 */
inline tcapint gpu_inner_size(const BaseTensor &t, const tcapint &axis) {
  tcapint inner = 1U;
  for (tcapint d = 0U; d < axis; ++d) {
    inner *= t.shape[d];
  }

  return inner;
}

/**
 * Write the elements of the (flat) tensor "src" to the corresponding elements
 * of "dst", which can have any strides. (This is the inverse of gpu_flat(),
 * for tensors that are modified in-place.)
 */
void gpu_scatter(Tensor &dst, const Tensor &src);

/**
 * For a tensor that a kernel modifies in-place: "flat" is what the kernel
 * operates on, and commit() writes it back, if it had to be a copy.
 */
struct GpuFlatDest {
  Tensor &orig;
  bool is_copy;
  Tensor flat;
  GpuFlatDest(Tensor &t)
      : orig(t), is_copy(!gpu_is_flat(t)), flat(gpu_flat(t)) {}
  void commit() {
    if (is_copy) {
      gpu_scatter(orig, flat);
    }
  }
};

/**
 * View "t" (for which every index has either the extent of "shape" or extent
 * 1) as broadcast to "shape", without copying.
 */
Tensor gpu_broadcast_view(const Tensor &t, const std::vector<tcapint> &shape);
} // namespace Weed

// Dispatch for element-wise kernels. The kernel arguments are, per operand, in
// order: (offset, step). Operands that the kernels can't address directly are
// gathered into contiguous copies first (and written back, for a tensor that
// is modified in-place).

// out = f(a), for kernel(a, out, args[, c])
#define GPU_FLAT_UNARY_C(type_a, type_o, api_call, c)                          \
  const Tensor fa = gpu_flat(a);                                               \
  gpu_require_flat(out, #api_call);                                            \
  const tcapint args[12U]{fa.offset, gpu_step(fa), out.offset, gpu_step(out),  \
                          0U,        0U,           0U,         0U,             \
                          0U,        0U,           0U,         0U};            \
  std::shared_ptr<type_a> a_storage =                                          \
      std::dynamic_pointer_cast<type_a>(fa.storage);                           \
  std::shared_ptr<type_o> o_storage =                                          \
      std::dynamic_pointer_cast<type_o>(out.storage);                          \
  a_storage->dev->RequestKernel(api_call, args, out.get_broadcast_size(),      \
                                {a_storage->buffer, o_storage->buffer}, 0U, c)
#define GPU_FLAT_UNARY(type_a, type_o, api_call)                               \
  GPU_FLAT_UNARY_C(type_a, type_o, api_call, nullptr)

// out = f(a, b), for kernel(a, b, out, args)
#define GPU_FLAT_BINARY(type_a, type_b, type_o, api_call)                      \
  const Tensor fa = gpu_flat(a);                                               \
  const Tensor fb = gpu_flat(b);                                               \
  gpu_require_flat(out, #api_call);                                            \
  const tcapint args[12U]{fa.offset,    gpu_step(fa), fb.offset,               \
                          gpu_step(fb), out.offset,   gpu_step(out),           \
                          0U,           0U,           0U,                      \
                          0U,           0U,           0U};                     \
  std::shared_ptr<type_a> a_storage =                                          \
      std::dynamic_pointer_cast<type_a>(fa.storage);                           \
  std::shared_ptr<type_b> b_storage =                                          \
      std::dynamic_pointer_cast<type_b>(fb.storage);                           \
  std::shared_ptr<type_o> o_storage =                                          \
      std::dynamic_pointer_cast<type_o>(out.storage);                          \
  a_storage->dev->RequestKernel(                                               \
      api_call, args, out.get_broadcast_size(),                                \
      {a_storage->buffer, b_storage->buffer, o_storage->buffer})

// a = f(a, b), in-place, for kernel(a, b, args)
#define GPU_FLAT_IN_PLACE(type_a, type_b, api_call)                            \
  GpuFlatDest da(a);                                                           \
  const Tensor fb = gpu_flat(b);                                               \
  const tcapint args[12U]{da.flat.offset,                                      \
                          gpu_step(da.flat),                                   \
                          fb.offset,                                           \
                          gpu_step(fb),                                        \
                          0U,                                                  \
                          0U,                                                  \
                          0U,                                                  \
                          0U,                                                  \
                          0U,                                                  \
                          0U,                                                  \
                          0U,                                                  \
                          0U};                                                 \
  std::shared_ptr<type_a> a_storage =                                          \
      std::dynamic_pointer_cast<type_a>(da.flat.storage);                      \
  std::shared_ptr<type_b> b_storage =                                          \
      std::dynamic_pointer_cast<type_b>(fb.storage);                           \
  a_storage->dev->RequestKernel(api_call, args, a.get_broadcast_size(),        \
                                {a_storage->buffer, b_storage->buffer});       \
  da.commit()

// din += f(in, dout), for kernel(din, in, dout, args[, c])
#define GPU_FLAT_GRAD_C(type_din, type_in, type_dout, api_call, c)             \
  GpuFlatDest fdin(din);                                                       \
  const Tensor fin = gpu_flat(in);                                             \
  const Tensor fdout = gpu_flat(dout);                                         \
  const tcapint args[12U]{fdin.flat.offset,                                    \
                          gpu_step(fdin.flat),                                 \
                          fin.offset,                                          \
                          gpu_step(fin),                                       \
                          fdout.offset,                                        \
                          gpu_step(fdout),                                     \
                          0U,                                                  \
                          0U,                                                  \
                          0U,                                                  \
                          0U,                                                  \
                          0U,                                                  \
                          0U};                                                 \
  std::shared_ptr<type_din> a_storage =                                        \
      std::dynamic_pointer_cast<type_din>(fdin.flat.storage);                  \
  std::shared_ptr<type_in> b_storage =                                         \
      std::dynamic_pointer_cast<type_in>(fin.storage);                         \
  std::shared_ptr<type_dout> c_storage =                                       \
      std::dynamic_pointer_cast<type_dout>(fdout.storage);                     \
  a_storage->dev->RequestKernel(                                               \
      api_call, args, din.get_broadcast_size(),                                \
      {a_storage->buffer, b_storage->buffer, c_storage->buffer}, 0U, c);       \
  fdin.commit()
#define GPU_FLAT_GRAD(type_din, type_in, type_dout, api_call)                  \
  GPU_FLAT_GRAD_C(type_din, type_in, type_dout, api_call, nullptr)
