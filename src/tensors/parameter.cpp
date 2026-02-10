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

#include "tensors/parameter.hpp"
#include "common/serializer.hpp"

namespace Weed {
void Parameter::save(std::ostream &out) const {
  Serializer::write_tcapint(out, offset);
  Serializer::write_tcapint(out, (tcapint)(shape.size()));
  for (size_t i = 0U; i < shape.size(); ++i) {
    Serializer::write_tcapint(out, shape[i]);
    Serializer::write_tcapint(out, stride[i]);
  }
  storage->save(out);
}
ParameterPtr Parameter::load(std::istream &in) {
  tcapint offset;
  Serializer::read_tcapint(in, offset);

  tcapint sz;
  Serializer::read_tcapint(in, sz);

  std::vector<tcapint> shape(sz);
  std::vector<tcapint> stride(sz);

  for (size_t i = 0U; i < shape.size(); ++i) {
    Serializer::read_tcapint(in, shape[i]);
    Serializer::read_tcapint(in, stride[i]);
  }

  StoragePtr storage = Storage::load(in);

  ParameterPtr p = std::make_shared<Parameter>(shape, stride, true,
                                               storage->dtype, storage->device);
  p->storage = storage;

  return p;
}
} // namespace Weed
