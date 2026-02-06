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

#include "modules/module.hpp"
#include "common/serializer.hpp"

#include "modules/sequential.hpp"

namespace Weed {
void Module::save(std::ostream &os) const {
  write_module_type(os, mtype);
  // Needs the inheriting struct to do the rest
}

ModulePtr Module::load(std::istream &is) {
  ModuleType mtype;
  read_module_type(is, mtype);

  switch (mtype) {
  case ModuleType::SEQUENTIAL_T: {
    tcapint sz;
    Serializer::read_tcapint(is, sz);
    std::vector<ModulePtr> mv;
    mv.reserve(sz);
    for (tcapint i = 0U; i < sz; ++i) {
      mv.push_back(load(is));
    }
    return std::make_shared<Sequential>(mv);
  }
  case ModuleType::NONE_MODULE_TYPE:
  default:
    throw std::domain_error("Can't recognize StorageType in Storage::load!");
  }

  return nullptr;
}
} // namespace Weed
