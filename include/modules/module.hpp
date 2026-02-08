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

#include "enums/module_type.hpp"
#include "tensors/parameter.hpp"

#if QRACK_AVAILABLE
#include "qrack/qfactory.hpp"
#endif

namespace Weed {
struct Module;
typedef std::shared_ptr<Module> ModulePtr;
/**
 * Composable module with forward function and parameters for autograd
 * optimization
 */
struct Module {
  ModuleType mtype;
  Module(ModuleType t) : mtype(t) {}
#if QRACK_AVAILABLE
  virtual TensorPtr forward(Qrack::QInterfacePtr q,
                            const std::vector<bitLenInt> &c,
                            const bitLenInt &t) {
    throw std::domain_error(
        "Only quantum Module instances apply forward() with QInterface!");
  }
  virtual TensorPtr forward(Qrack::QInterfacePtr q) {
    throw std::domain_error(
        "Only quantum Module instances apply forward() with QInterface!");
  }
  virtual TensorPtr forward() {
    throw std::domain_error(
        "Only quantum Module instances apply forward() with no parameters!");
  }
#endif
  virtual TensorPtr forward(const TensorPtr) = 0;
  virtual TensorPtr forward(const BaseTensorPtr t) {
    return forward(std::dynamic_pointer_cast<Tensor>(t));
  }
  virtual std::vector<ParameterPtr> parameters() {
    return std::vector<ParameterPtr>();
  }
  virtual void train() {
    std::vector<ParameterPtr> params = parameters();
    for (const auto &p : params) {
      p->train();
    }
  }
  virtual void eval() {
    std::vector<ParameterPtr> params = parameters();
    for (const auto &p : params) {
      p->eval();
    }
  }
  virtual ~Module() {}
  /**
   * Serialize storage to ostream
   */
  virtual void save(std::ostream &) const;
  /**
   * Load serialized storage from istream
   */
  static ModulePtr load(std::istream &);
  static void write_module_type(std::ostream &out, const ModuleType &x) {
    out.write(reinterpret_cast<const char *>(&x), sizeof(ModuleType));
  }
  static void read_module_type(std::istream &in, ModuleType &x) {
    in.read(reinterpret_cast<char *>(&x), sizeof(ModuleType));
  }
};
} // namespace Weed
