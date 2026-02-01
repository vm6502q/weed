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
int benchmarkSamples = 100;
int benchmarkDepth = -1;
DeviceTag TEST_DTAG;

#if ENABLE_OPENCL
#define WEED_GPU_SINGLETON (OCLEngine::Instance())
#elif ENABLE_CUDA
#define WEED_GPU_SINGLETON (CUDAEngine::Instance())
#endif

#define SHOW_OCL_BANNER() WEED_GPU_SINGLETON.GetDeviceCount();

int main(int argc, char *argv[]) {
  Catch::Session session;

  // Engines
  bool cpu = false;
  bool gpu = false;

  using namespace Catch::clara;

  /*
   * Allow specific layers and processor types to be enabled.
   */
  auto cli =
      session.cli() |
      Opt(cpu)["--device-cpu"]("Enable the CPU-based implementation tests") |
      Opt(gpu)["--device-gpu"]("Single (parallel) processor GPU tests") |
      Opt(async_time)["--async-time"]("Time based on asynchronous return") |
      Opt(device_id, "device-id")["-d"]["--device-id"](
          "GPU device ID (\"-1\" for default device)") |
      Opt(sparse)["--sparse"](
          "(For QEngineCPU, under QUnit:) Use a state vector optimized for "
          "sparse representation and iteration.") |
      Opt(benchmarkSamples, "samples")["--samples"](
          "number of samples to collect (default: 100)") |
      Opt(benchmarkDepth, "depth")["--benchmark-depth"](
          "depth of randomly constructed circuits, when applicable, with 1 "
          "round of single qubit and 1 round of "
          "multi-qubit gates being 1 unit of depth (default: 0, for square "
          "circuits)");

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

  session.config().stream()
      << "Random Seed: " << session.configData().rngSeed << std::endl;

  if (!cpu && !gpu) {
    cpu = true;
    gpu = true;
  }

#if ENABLE_OPENCL || ENABLE_CUDA
  SHOW_OCL_BANNER();
#endif

  int num_failed = 0;

  session.config().stream() << std::endl;

  if (num_failed == 0 && cpu) {
    session.config().stream() << "##################################### CPU "
                                 "#####################################"
                              << std::endl;
    TEST_DTAG = DeviceTag::CPU;
    num_failed = session.run();
  }

#if ENABLE_GPU
  if (num_failed == 0 && gpu) {
    session.config().stream() << "##################################### GPU "
                                 "#####################################"
                              << std::endl;
    TEST_DTAG = DeviceTag::GPU;
    num_failed = session.run();
  }
#endif

  return num_failed;
}
