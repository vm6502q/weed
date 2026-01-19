//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "tensor.hpp"

namespace Weed {
struct MulKernel {
    void (*cpu)(const Tensor&, const Tensor&, Tensor&);
    void (*opencl)(const Tensor&, const Tensor&, Tensor&);
};

extern MulKernel mul_kernel;

void mul(const Tensor& a, const Tensor& b, Tensor& out) {
    switch (out.storage->device) {
        case DeviceTag::CPU:
            mul_kernel.cpu(a, b, out);
            break;
        case DeviceTag::OpenCL:
            mul_kernel.opencl(a, b, out);
            break;
    }
}
} // namespace Weed
