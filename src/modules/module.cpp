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

#include "modules/linear.hpp"
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
  case ModuleType::LINEAR_T: {
    tcapint in_features, out_features;
    Serializer::read_tcapint(is, in_features);
    Serializer::read_tcapint(is, out_features);
    ParameterPtr weight = Parameter::load(is);
    bool is_bias;
    Serializer::read_bool(is, is_bias);
    ParameterPtr bias = nullptr;
    if (is_bias) {
      bias = Parameter::load(is);
    }
    LinearPtr l = std::make_shared<Linear>();
    l->in_features = in_features;
    l->out_features = out_features;
    l->weight = weight;
    l->bias = bias;

    return l;
  }
  case ModuleType::NONE_MODULE_TYPE:
  default:
    throw std::domain_error("Can't recognize StorageType in Storage::load!");
  }

  return nullptr;
}
} // namespace Weed
