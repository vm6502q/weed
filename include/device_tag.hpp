//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

namespace Weed {
enum DeviceTag {
    CPU = 1,
    OpenCL = 2,
    Qrack = 3,
    CUDA = 4,
    DEFAULT = CPU
};
} // namespace Weed
