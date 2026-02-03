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

#include "storage/int_storage.hpp"

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
#if ENABLE_ENV_VARS
const tlenint PSTRIDEPOW_DEFAULT =
    (tlenint)(getenv("WEED_PSTRIDEPOW")
                  ? std::stoi(std::string(getenv("WEED_PSTRIDEPOW")))
                  : PSTRIDEPOW);
const tcapint GSTRIDE =
    (tcapint)(getenv("WEED_GSTRIDE")
                  ? std::stoi(std::string(getenv("WEED_GSTRIDE")))
                  : ((1 << PSTRIDEPOW_DEFAULT) *
                     std::thread::hardware_concurrency()));
#else
const tlenint PSTRIDEPOW_DEFAULT = PSTRIDEPOW;
const tcapint GSTRIDE =
    (1 << PSTRIDEPOW_DEFAULT) * std::thread::hardware_concurrency();
#endif
#else
// Never auto-switch to a GPU if we don't have one.
const tcapint GSTRIDE = -1;
#endif

SymbolTensor::SymbolTensor(const std::vector<tcapint> &shp,
                           const std::vector<tcapint> &strd, const bool &rg,
                           const DeviceTag &_dtag, const int64_t &did,
                           const bool &s)
    : BaseTensor(shp, strd), grad_node(nullptr), requires_grad(rg),
      freeze(shp.size(), false) {

  const tcapint size = get_size();
  DeviceTag dtag = _dtag;
  if (dtag == DEFAULT_DEVICE) {
    if (size > GSTRIDE) {
      dtag = DeviceTag::GPU;
    } else {
      dtag = DeviceTag::CPU;
    }
  }

  if (s && (dtag == DeviceTag::CPU)) {
    storage = std::make_shared<SparseCpuIntStorage>(size);
  } else {
#if ENABLE_GPU
    INIT_DEVICE_STORAGE(size, GpuIntStorage, CpuIntStorage);
#else
    storage = std::make_shared<CpuIntStorage>(size);
#endif
  }
}

SymbolTensor::SymbolTensor(const std::vector<tcapint> &val,
                           const std::vector<tcapint> &shp,
                           const std::vector<tcapint> &strd, const bool &rg,
                           const DeviceTag &_dtag, const int64_t &did)
    : BaseTensor(shp, strd), grad_node(nullptr), requires_grad(rg),
      freeze(shp.size(), false) {

  const tcapint size = get_size();

  if (size != val.size()) {
    throw std::invalid_argument("Tensor value initializer vector must have "
                                "same size as implied by shape and stride!");
  }

  DeviceTag dtag = _dtag;
  if (dtag == DEFAULT_DEVICE) {
    if (size > GSTRIDE) {
      dtag = DeviceTag::GPU;
    } else {
      dtag = DeviceTag::CPU;
    }
  }

#if ENABLE_GPU
  INIT_DEVICE_STORAGE(val, GpuIntStorage, CpuIntStorage);
#else
  storage = std::make_shared<CpuIntStorage>(val);
#endif
}

Tensor::Tensor(const IntSparseVector &val, const std::vector<tcapint> &shp,
               const std::vector<tcapint> &strd, const bool &rg)
    : BaseTensor(shp, strd), grad_node(nullptr), requires_grad(rg),
      freeze(shp.size(), false) {

  storage = std::make_shared<SparseCpuIntStorage>(val, get_size());
}
} // namespace Weed
