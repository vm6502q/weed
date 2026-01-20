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

#include "tensor.hpp"

#include <chrono>
#include <iostream>

#include "catch.hpp"

#include "tests.hpp"

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <direct.h>
#endif

#define EPSILON 0.001
#define REQUIRE_FLOAT(A, B)                                                                                            \
    do {                                                                                                               \
        real1_f __tmp_a = A;                                                                                           \
        real1_f __tmp_b = B;                                                                                           \
        REQUIRE(__tmp_a < (__tmp_b + EPSILON));                                                                        \
        REQUIRE(__tmp_b > (__tmp_b - EPSILON));                                                                        \
    } while (0);

using namespace Weed;

const double clockFactor = 1.0 / 1000.0; // Report in ms

double formatTime(double t, bool logNormal)
{
    if (logNormal) {
        return pow(2.0, t);
    } else {
        return t;
    }
}

void benchmarkLoopVariable(std::function<void(Tensor, bitLenInt)> fn, bitLenInt mxQbts, bool logNormal = false)
{
    std::cout << std::endl;
    std::cout << ">>> '" << Catch::getResultCapture().getCurrentTestName() << "':" << std::endl;
    std::cout << benchmarkSamples << " iterations" << std::endl;
    std::cout << "# of Qubits, ";
    std::cout << "Average Time (ms), ";
    std::cout << "Sample Std. Deviation (ms), ";
    std::cout << "Fastest (ms), ";
    std::cout << "1st Quartile (ms), ";
    std::cout << "Median (ms), ";
    std::cout << "3rd Quartile (ms), ";
    std::cout << "Slowest (ms), ";
    std::cout << "Failure count, ";
    std::cout << "Average SDRP Fidelity" << std::endl;

    std::vector<double> trialClocks;

    bitLenInt mnQbts;
    if (single_qubit_run) {
        mnQbts = mxQbts;
    } else {
        mnQbts = min_qubits;
    }

    for (bitLenInt numBits = mnQbts; numBits <= mxQbts; numBits++) {
        // QInterfacePtr qftReg = CreateQuantumInterface(engineStack, numBits, ZERO_BCI, rng, CMPLX_DEFAULT_ARG,
        //     enable_normalization, true, use_host_dma, device_id, !disable_hardware_rng, sparse, REAL1_EPSILON, devList);
        double avgt = 0.0;
        double avgf = 0.0;
        int sampleFailureCount = 0;
        trialClocks.clear();

        std::vector<bitCapInt> qPowers;
        for (bitLenInt i = 0U; i < numBits; ++i) {
            qPowers.push_back(pow2(i));
        }

        for (int sample = 0; sample < benchmarkSamples; sample++) {}

        avgt /= trialClocks.size();
        avgf /= trialClocks.size();

        double stdet = 0.0;
        for (int sample = 0; sample < (int)trialClocks.size(); sample++) {
            stdet += (trialClocks[sample] - avgt) * (trialClocks[sample] - avgt);
        }
        stdet = sqrt(stdet / trialClocks.size());

        std::sort(trialClocks.begin(), trialClocks.end());

        std::cout << (int)numBits << ", "; /* # of Qubits */
        std::cout << formatTime(avgt, logNormal) << ","; /* Average Time (ms) */
        std::cout << formatTime(stdet, logNormal) << ","; /* Sample Std. Deviation (ms) */

        // Fastest (ms)
        std::cout << formatTime(trialClocks[0], logNormal) << ",";

        if (trialClocks.size() == 1) {
            std::cout << formatTime(trialClocks[0], logNormal) << ",";
            std::cout << formatTime(trialClocks[0], logNormal) << ",";
            std::cout << formatTime(trialClocks[0], logNormal) << ",";
            std::cout << formatTime(trialClocks[0], logNormal) << ",";
            std::cout << sampleFailureCount << ",";
            std::cout << avgf << std::endl;
            continue;
        }

        // 1st Quartile (ms)
        if (trialClocks.size() < 8) {
            std::cout << formatTime(trialClocks[0], logNormal) << ",";
        } else if (trialClocks.size() % 4 == 0) {
            std::cout << formatTime((trialClocks[trialClocks.size() / 4 - 1] + trialClocks[trialClocks.size() / 4]) / 2,
                             logNormal)
                      << ",";
        } else {
            std::cout << formatTime(trialClocks[trialClocks.size() / 4 - 1] / 2, logNormal) << ",";
        }

        // Median (ms) (2nd quartile)
        if (trialClocks.size() < 4) {
            std::cout << formatTime(trialClocks[trialClocks.size() / 2], logNormal) << ",";
        } else if (trialClocks.size() % 2 == 0) {
            std::cout << formatTime((trialClocks[trialClocks.size() / 2 - 1] + trialClocks[trialClocks.size() / 2]) / 2,
                             logNormal)
                      << ",";
        } else {
            std::cout << formatTime(trialClocks[trialClocks.size() / 2 - 1] / 2, logNormal) << ","; /* Median (ms) */
        }

        // 3rd Quartile (ms)
        if (trialClocks.size() < 8) {
            std::cout << formatTime(trialClocks[(3 * trialClocks.size()) / 4], logNormal) << ",";
        } else if (trialClocks.size() % 4 == 0) {
            std::cout << formatTime((trialClocks[(3 * trialClocks.size()) / 4 - 1] +
                                        trialClocks[(3 * trialClocks.size()) / 4]) /
                                 2,
                             logNormal)
                      << ",";
        } else {
            std::cout << formatTime(trialClocks[(3 * trialClocks.size()) / 4 - 1] / 2, logNormal) << ",";
        }

        // Slowest (ms)
        if (trialClocks.size() <= 1) {
            std::cout << formatTime(trialClocks[0], logNormal) << ",";
        } else {
            std::cout << formatTime(trialClocks[trialClocks.size() - 1], logNormal) << ",";
        }

        // Failure count
        std::cout << sampleFailureCount << ",";
        // Average SDRP fidelity
        std::cout << avgf << std::endl;
    }
}

void benchmarkLoop(std::function<void(Tensor, bitLenInt)> fn, bool logNormal = false)
{
    benchmarkLoopVariable(fn, max_qubits, logNormal);
}

TEST_CASE("test_default", "[suite]")
{
    benchmarkLoop([](Tensor tensor, bitLenInt n) {});
}
