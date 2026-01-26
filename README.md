<img width="1536" height="1024" alt="weed_logo" src="https://github.com/vm6502q/weed/blob/main/weed_logo.png" />

# Weed
Minimalist AI/ML inference and backprogation in the style of [Qrack](https://github.com/unitaryfoundation/qrack)

## Development Status
**Weed** is a rapidly-developing **work-in-progress**. Its ABI may change drastically and without notice.

The project provides a set of essential CPU and GPU **kernels**, used by `Tensor` instances that perform _autograd._ We also provide _stochastic gradient descent (SGD)_ and _Adam_ optimizer implementations. (Build and check the API reference to get started.)

## Building the API reference

```sh
    $ doxygen doxygen.config
```

## Performing code coverage

```sh
    $ cd _build
    $ cmake -DENABLE_CODECOVERAGE=ON ..
    $ make -j 8 unittest
    $ ./unittest
    $ make coverage
    $ cd coverage_results
    $ python -m http.server
```

## Copyright, License, and Acknowledgments

Copyright (c) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.

The Weed logo was produced with assistance from "Elara," an OpenAI custom GPT, and it is in the **public domain**.

Licensed under the GNU Lesser General Public License V3.

See [LICENSE.md](https://github.com/vm6502q/qrack/blob/main/LICENSE.md) in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html for details.
