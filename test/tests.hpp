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
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#pragma once

#include "device_tag.hpp"
#include "parameter.hpp"

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
extern int benchmarkSamples;
extern int benchmarkDepth;
extern Weed::DeviceTag TEST_DTAG;

#include "catch.hpp"
