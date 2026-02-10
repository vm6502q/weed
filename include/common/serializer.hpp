//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of
// Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#pragma once

#include "common/weed_types.hpp"
#include "enums/device_tag.hpp"
#include "enums/dtype.hpp"
#include "enums/quantum_function_type.hpp"
#include "enums/storage_type.hpp"
#include "qrack/qneuron.hpp"

namespace Weed {
/**
 * Static methods for serialization and de-serialization
 */
struct Serializer {
  static void write_bool(std::ostream &out, const bool &x) {
    out.write(reinterpret_cast<const char *>(&x), sizeof(bool));
  }
  static void read_bool(std::istream &in, bool &x) {
    in.read(reinterpret_cast<char *>(&x), sizeof(bool));
  }
  static void write_tcapint(std::ostream &out, const tcapint &x) {
    out.write(reinterpret_cast<const char *>(&x), sizeof(tcapint));
  }
  static void read_tcapint(std::istream &in, tcapint &x) {
    in.read(reinterpret_cast<char *>(&x), sizeof(tcapint));
  }
  static void write_symint(std::ostream &out, const symint &x) {
    out.write(reinterpret_cast<const char *>(&x), sizeof(symint));
  }
  static void read_symint(std::istream &in, symint &x) {
    in.read(reinterpret_cast<char *>(&x), sizeof(symint));
  }
  static void write_real(std::ostream &out, const real1 &x) {
    out.write(reinterpret_cast<const char *>(&x), sizeof(real1));
  }
  static void read_real(std::istream &in, real1 &x) {
    in.read(reinterpret_cast<char *>(&x), sizeof(real1));
  }
  static void write_complex(std::ostream &out, const complex &z) {
    write_real(out, z.real());
    write_real(out, z.imag());
  }
  static void read_complex(std::istream &in, complex &z) {
    real1 r, i;
    read_real(in, r);
    read_real(in, i);
    z = complex(r, i);
  }
  static void write_quantum_fn(std::ostream &out,
                               const QuantumFunctionType &x) {
    out.write(reinterpret_cast<const char *>(&x), sizeof(QuantumFunctionType));
  }
  static void read_quantum_fn(std::istream &in, QuantumFunctionType &x) {
    in.read(reinterpret_cast<char *>(&x), sizeof(QuantumFunctionType));
  }
  static void write_qneuron_activation_fn(std::ostream &out,
                                          const Qrack::QNeuronActivationFn &x) {
    out.write(reinterpret_cast<const char *>(&x),
              sizeof(Qrack::QNeuronActivationFn));
  }
  static void read_qneuron_activation_fn(std::istream &in,
                                         Qrack::QNeuronActivationFn &x) {
    in.read(reinterpret_cast<char *>(&x), sizeof(Qrack::QNeuronActivationFn));
  }
};
} // namespace Weed
