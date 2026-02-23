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

#include "tensors/base_tensor.hpp"

#include <thread>

#define GET_REAL(ptr) static_cast<RealScalar *>((ptr).get())->get_item()
#define IS_SPARSE(a)                                                           \
  (a && a->storage->is_sparse() &&                                             \
   ((a->storage->get_sparse_size() << 1U) < a->storage->size))

#if ENABLE_GPU
#define INIT_DEVICE_STORAGE(val, GpuType, CpuType)                             \
  switch (dtag) {                                                              \
  case DeviceTag::GPU:                                                         \
    storage = std::make_shared<GpuType>(val, did);                             \
    break;                                                                     \
  case DeviceTag::CPU:                                                         \
  default:                                                                     \
    storage = std::make_shared<CpuType>(val);                                  \
  }
#else
// There is only one device type available:
#define INIT_DEVICE_STORAGE(val, GpuType, CpuType)                             \
  storage = std::make_shared<CpuType>(val);
#endif

namespace Weed {
#if ENABLE_GPU
#if WEED_ENABLE_ENV_VARS
const tlenint PSTRIDEPOW_DEFAULT =
    (tlenint)(getenv("WEED_PSTRIDEPOW")
                  ? std::stoi(std::string(getenv("WEED_PSTRIDEPOW")))
                  : WEED_PSTRIDEPOW);
const tcapint GSTRIDE =
    (tcapint)(getenv("WEED_GSTRIDE")
                  ? std::stoi(std::string(getenv("WEED_GSTRIDE")))
                  : ((1 << PSTRIDEPOW_DEFAULT) *
                     std::thread::hardware_concurrency()));
#else
const tlenint PSTRIDEPOW_DEFAULT = WEED_PSTRIDEPOW;
const tcapint GSTRIDE =
    (1 << PSTRIDEPOW_DEFAULT) * std::thread::hardware_concurrency();
#endif
#else
// Never auto-switch to a GPU if we don't have one.
const tcapint GSTRIDE = -1;
#endif

DeviceTag
BaseTensor::get_dtag_by_presidence(const std::vector<BaseTensorPtr> &v) {
#if ENABLE_GPU
  for (const BaseTensorPtr &p : v) {
    const tcapint sz = p->storage->size;
    const tcapint sp = p->storage->get_sparse_size();
    if (sz == sp) {
      if (sz > GSTRIDE) {
        return DeviceTag::GPU;
      }
    } else {
      if ((sp << 1U) > GSTRIDE) {
        return DeviceTag::GPU;
      }
    }
  }
#endif

  return DeviceTag::CPU;
}
} // namespace Weed
