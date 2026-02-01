//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.en.html for details.

#include <iostream>

#include "catch.hpp"

#include "tests.hpp"

#include <chrono>

#include "common/weed_functions.hpp"
#include "tensors/flat_tensors.hpp"

const double clock_factor = 1.0 / 1000.0; // Report in ms

using namespace Weed;

TEST_CASE("test_random_real_mul") {
  std::cout << "# of elements (power of 2), Time (ms)" << std::endl;

  for (tlenint b = 10U; b < 24U; ++b) {
    const tcapint p = pow2Gpu(b);

    TensorPtr x = std::make_shared<Tensor>(std::vector<tcapint>{p},
                                           std::vector<tcapint>{1U}, false,
                                           DType::REAL, TEST_DTAG, -1, sparse);
    x->storage->FillOnes();
    TensorPtr y = std::make_shared<Tensor>(std::vector<tcapint>{p},
                                           std::vector<tcapint>{1U}, false,
                                           DType::REAL, TEST_DTAG, -1, sparse);
    y->storage->FillOnes();

    const auto start = std::chrono::high_resolution_clock::now();
    TensorPtr z = x * y;
    const auto end = std::chrono::high_resolution_clock::now();

    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << (int)b << ", " << (clock_factor * time.count()) << std::endl;
  }
}
