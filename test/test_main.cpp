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

#define CATCH_CONFIG_RUNNER /* Access to the configuration. */
#include "tests.hpp"

#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

using namespace Weed;

bool use_host_dma = false;
bool async_time = false;
bool sparse = false;
int device_id = -1;
bitLenInt max_qubits = 24;
int benchmarkSamples = 100;
int benchmarkDepth = -1;
std::vector<int64_t> devList;

#if ENABLE_OPENCL
#define WEED_GPU_SINGLETON (OCLEngine::Instance())
#define WEED_GPU_CLASS QEngineOCL
#define WEED_GPU_ENUM QINTERFACE_OPENCL
#elif ENABLE_CUDA
#define WEED_GPU_SINGLETON (CUDAEngine::Instance())
#define WEED_GPU_CLASS QEngineCUDA
#define WEED_GPU_ENUM QINTERFACE_CUDA
#endif
#define SHOW_OCL_BANNER()                                                                                              \
    if (WEED_GPU_SINGLETON.GetDeviceCount()) {                                                                        \
        CreateQuantumInterface(WEED_GPU_ENUM, 1, ZERO_BCI).reset();                                                   \
    }

int main(int argc, char* argv[])
{
    Catch::Session session;

    // Engines
    bool cpu = false;
    bool opencl = false;
    bool hybrid = false;
    bool cuda = false;

    std::string devListStr;

    using namespace Catch::clara;

    /*
     * Allow specific layers and processor types to be enabled.
     */
    auto cli = session.cli() | Opt(cpu)["--proc-cpu"]("Enable the CPU-based implementation tests") |
        Opt(opencl)["--proc-opencl"]("Single (parallel) processor OpenCL tests") |
        Opt(hybrid)["--proc-hybrid"]("Enable CPU/OpenCL hybrid implementation tests") |
        Opt(cuda)["--proc-cuda"]("Enable QEngineCUDA tests") |
        Opt(async_time)["--async-time"]("Time based on asynchronous return") |
        Opt(device_id, "device-id")["-d"]["--device-id"]("Opencl device ID (\"-1\" for default device)") |
        Opt(sparse)["--sparse"](
            "(For QEngineCPU, under QUnit:) Use a state vector optimized for sparse representation and iteration.") |
        Opt(benchmarkSamples, "samples")["--samples"]("number of samples to collect (default: 100)") |
        Opt(benchmarkDepth, "depth")["--benchmark-depth"](
            "depth of randomly constructed circuits, when applicable, with 1 round of single qubit and 1 round of "
            "multi-qubit gates being 1 unit of depth (default: 0, for square circuits)") |
        Opt(devListStr, "devices")["--devices"](
            "list of devices, for QPager (default is solely default OpenCL device)");

    session.cli(cli);

    /* Set some defaults for convenience. */
    session.configData().useColour = Catch::UseColour::No;
    session.configData().rngSeed = std::time(0);

    // session.configData().abortAfter = 1;

    /* Parse the command line. */
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) {
        return returnCode;
    }

    session.config().stream() << "Random Seed: " << session.configData().rngSeed;

    if (disable_hardware_rng) {
        session.config().stream() << std::endl;
    } else {
        session.config().stream() << " (Overridden by hardware generation!)" << std::endl;
    }

    if (!cpu && !opencl && !hybrid && !cuda) {
        cpu = true;
        opencl = true;
        cuda = true;
        hybrid = true;
    }

    if (devListStr.compare("") != 0) {
        std::stringstream devListStr_stream(devListStr);
        while (devListStr_stream.good()) {
            std::string substr;
            getline(devListStr_stream, substr, ',');
            devList.push_back(stoi(substr));
        }
    }

// #if ENABLE_OPENCL || ENABLE_CUDA
//     SHOW_OCL_BANNER();
// #endif

    int num_failed = 0;

    if (num_failed == 0 && cpu) {
        session.config().stream() << "############ QEngine -> CPU ############" << std::endl;
        num_failed = session.run();
    }

#if ENABLE_OPENCL
    if (num_failed == 0 && opencl) {
        session.config().stream() << "############ QEngine -> OpenCL ############" << std::endl;
        num_failed = session.run();
    }
#endif

#if ENABLE_CUDA
    if (num_failed == 0 && cuda) {
        session.config().stream() << "############ QEngine -> CUDA ############" << std::endl;
        num_failed = session.run();
    }
#endif

#if ENABLE_OPENCL || ENABLE_CUDA
    if (num_failed == 0 && hybrid) {
        session.config().stream() << "############ QUnit -> QHybrid ############" << std::endl;
        num_failed = session.run();
    }
#endif

    return num_failed;
}

QInterfaceTestFixture::QInterfaceTestFixture()
{
    uint32_t rngSeed = Catch::getCurrentContext().getConfig()->rngSeed();

    std::cout << ">>> '" << Catch::getResultCapture().getCurrentTestName() << "':" << std::endl;
}
