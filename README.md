<img width="1536" height="1024" alt="weed_logo" src="https://github.com/vm6502q/weed/blob/main/weed_logo.png" />

# Weed
Minimalist AI/ML inference and backprogation in the style of [Qrack](https://github.com/unitaryfoundation/qrack)

## Development Status
**Weed** is a rapidly-developing **work-in-progress**. Its ABI may change drastically and without notice.

The project provides a set of essential CPU and GPU **kernels**, used by `Tensor` instances that perform _autograd._ We also provide _stochastic gradient descent (SGD)_ and _Adam_ optimizer implementations. (Build and check the API reference to get started.)

## Why try Weed?

With the growing popularity of AI/ML tools and workflows (including LLMs), legacy frameworks often carry "code debt" from over a decade of rapidly developing research history. This has led them to "bolt on" new features and advancements to design principles decided before the latest research. Popular frameworks also commonly started based in Python (maybe to capture early adoption), only later potentially "tacking on" a C++ library for special-case deployment needs. These conditions have produced libraries and frameworks with complicated dependency trees that occupy upward of a GB of disk footprint. This entire ecosystem might be due for a "refresh."

**Weed** does not seek to fully replace or supplant established frameworks. However, it aims for **minimalist complete closure** on the primitives necessary for high-performance AI/ML inference and back-propagation. Chiefly, this includes **kernels**, and a `Tensor` interface that immediately produces an **autograd** graph appropriate for training. Allowing **optional** OpenCL for **hardware acceleration**, it will remain **free of required dependencies** outside of C++(11) language standard.

Rethinking AI/ML library design this way, `Weed` has realized a rather unique and powerful form of _sparsification_ of `Tensor` **storage**. _Sparseness_ should **not** be a **`Tensor` interface concern**, but rather a **`Storage` concern**. Inspired by the design of the [Qrack](https://github.com/unitaryfoundation/qrack) quantum computer simulation framework, the `Tensor` interface treats **sparse and dense** tensors as **functionally equivalent**. Sparse optimization is so "transparently streamlined," this way, that it defaults to enabled for CPU-based tensors, and we recommend you leave it enabled at all times.

Much like `Qrack`, `Weed` is designed to make the correct thing the default—and the expensive thing explicit.

## Building the API reference

```sh
    $ doxygen doxygen.config
```

## Performing code coverage

```sh
    $ cd _build
    $ cmake -DENABLE_CODECOVERAGE=ON ..
    $ make unittest
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
