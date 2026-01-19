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
struct AddKernel {
    void (*cpu)(const Tensor&, const Tensor&, Tensor&);
    void (*opencl)(const Tensor&, const Tensor&, Tensor&);
};

extern AddKernel add_kernel;

void add(const Tensor& a, const Tensor& b, Tensor& out) {
    switch (out.storage->device) {
        case DeviceTag::CPU:
            add_kernel.cpu(a, b, out);
            break;
        case DeviceTag::OpenCL:
            add_kernel.opencl(a, b, out);
            break;
    }
}
} // namespace Weed
