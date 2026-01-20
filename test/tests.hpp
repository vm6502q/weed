//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2021. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "tensor.hpp"

/* A quick-and-dirty epsilon for clamping floating point values. */
#define QRACK_TEST_EPSILON 0.5

/*
 * Default engine type to run the tests with. Global because catch doesn't
 * support parameterization.
 */
extern bool use_host_dma;
extern bool async_time;
extern bool sparse;
extern int device_id;
extern bitLenInt max_qubits;
extern bitLenInt min_qubits;
extern bool single_qubit_run;
extern int benchmarkSamples;
extern int benchmarkDepth;
extern int timeout;
extern std::vector<int64_t> devList;
extern bool optimal;
extern bool optimal_single;

/* Declare the stream-to-probability prior to including catch.hpp. */
namespace Weed {
// TODO: Output functions
} // namespace Qrack

#include "catch.hpp"

/*
 * A fixture to create a unique QInterface test, of the appropriate type, for
 * each executing test case.
 */
class TensorTestFixture {
protected:
    Weed::Tensor tensor;

public:
    TensorTestFixture();
};
