//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "device_tag.hpp"
#include "dtype.hpp"
#include "common/weed_types.hpp"

namespace Weed {
struct Storage {
    DeviceTag device;
    DType dtype;
    vecCapIntGpu size;

    virtual ~Storage() {}
};
} // namespace Weed
