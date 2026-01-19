//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "storage.hpp"

namespace Weed {
struct Tensor {
    StoragePtr storage;

    std::vector<vecCapIntGpu> shape;
    std::vector<vecCapIntGpu> stride;
    vecCapIntGpu offset;

    bool requires_grad;
    struct Node* grad_node;

    Tensor()
        : offset(0U), requires_grad(false), grad_node(nullptr) {}
};
} // namespace Weed
