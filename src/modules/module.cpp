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

#include "modules/dropout.hpp"
#include "modules/embedding.hpp"
#include "modules/gru.hpp"
#include "modules/layernorm.hpp"
#include "modules/linear.hpp"
#include "modules/logsoftmax.hpp"
#include "modules/lstm.hpp"
#include "modules/migrate_cpu.hpp"
#include "modules/migrate_gpu.hpp"
#include "modules/relu.hpp"
#include "modules/sequential.hpp"
#include "modules/sigmoid.hpp"
#include "modules/softmax.hpp"
#include "modules/tanh.hpp"

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
    LinearPtr l = std::make_shared<Linear>();
    Serializer::read_tcapint(is, l->in_features);
    Serializer::read_tcapint(is, l->out_features);
    l->weight = Parameter::load(is);
    bool is_bias;
    Serializer::read_bool(is, is_bias);
    if (is_bias) {
      l->bias = Parameter::load(is);
    }

    return l;
  }
  case ModuleType::RELU_T: {
    return std::make_shared<ReLU>();
  }
  case ModuleType::SIGMOID_T: {
    return std::make_shared<Sigmoid>();
  }
  case ModuleType::TANH_T: {
    return std::make_shared<Tanh>();
  }
  case ModuleType::DROPOUT_T: {
    DropoutPtr d = std::make_shared<Dropout>();
    Serializer::read_real(is, d->p);
    Serializer::read_bool(is, d->training);

    return d;
  }
  case ModuleType::EMBEDDING_T: {
    EmbeddingPtr e = std::make_shared<Embedding>();
    Serializer::read_tcapint(is, e->num_embeddings);
    Serializer::read_tcapint(is, e->embedding_dim);
    e->weight = Parameter::load(is);

    return e;
  }
  case ModuleType::LAYERNORM_T: {
    LayerNormPtr l = std::make_shared<LayerNorm>();
    Serializer::read_tcapint(is, l->features);
    Serializer::read_real(is, l->eps);
    l->gamma = Parameter::load(is);
    l->beta = Parameter::load(is);

    return l;
  }
  case ModuleType::GRU_T: {
    GRUPtr g = std::make_shared<GRU>();
    Serializer::read_tcapint(is, g->input_dim);
    Serializer::read_tcapint(is, g->hidden_dim);
    g->W_x = std::dynamic_pointer_cast<Linear>(Linear::load(is));
    g->W_h = std::dynamic_pointer_cast<Linear>(Linear::load(is));
    g->state = Tensor::zeros({g->hidden_dim});

    return g;
  }
  case ModuleType::LSTM_T: {
    LSTMPtr l = std::make_shared<LSTM>();
    Serializer::read_tcapint(is, l->input_dim);
    Serializer::read_tcapint(is, l->hidden_dim);
    l->W_x = std::dynamic_pointer_cast<Linear>(Linear::load(is));
    l->W_h = std::dynamic_pointer_cast<Linear>(Linear::load(is));
    l->state = LSTMState{Tensor::zeros(std::vector<tcapint>{l->hidden_dim}),
                         Tensor::zeros(std::vector<tcapint>{l->hidden_dim})};

    return l;
  }
  case ModuleType::MIGRATE_CPU: {
    return std::make_shared<MigrateCpu>();
  }
  case ModuleType::MIGRATE_GPU: {
    return std::make_shared<MigrateGpu>();
  }
  case ModuleType::SOFTMAX: {
    tcapint axis;
    Serializer::read_tcapint(is, axis);
    return std::make_shared<Softmax>(axis);
  }
  case ModuleType::LOGSOFTMAX: {
    tcapint axis;
    Serializer::read_tcapint(is, axis);
    return std::make_shared<LogSoftmax>(axis);
  }
  case ModuleType::NONE_MODULE_TYPE:
  default:
    throw std::domain_error("Can't recognize ModuleType in Module::load!");
  }

  return nullptr;
}
} // namespace Weed
