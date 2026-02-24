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
#include "modules/flatten.hpp"
#include "modules/gelu.hpp"
#include "modules/gru.hpp"
#include "modules/layernorm.hpp"
#include "modules/learned_positional_encoding.hpp"
#include "modules/linear.hpp"
#include "modules/logsoftmax.hpp"
#include "modules/lstm.hpp"
#include "modules/max.hpp"
#include "modules/mean.hpp"
#include "modules/mean_center.hpp"
#include "modules/migrate_cpu.hpp"
#include "modules/migrate_gpu.hpp"
#include "modules/min.hpp"
#include "modules/multihead_attention.hpp"
#include "modules/positional_encoding.hpp"
#include "modules/relu.hpp"
#include "modules/reshape.hpp"
#include "modules/sequential.hpp"
#include "modules/sigmoid.hpp"
#include "modules/softmax.hpp"
#include "modules/stddev.hpp"
#include "modules/tanh.hpp"
#include "modules/transformer_encoder_layer.hpp"
#include "modules/variance.hpp"

#if QRACK_AVAILABLE
#include "modules/qrack_neuron_layer.hpp"
#include "storage/cpu_real_storage.hpp"
#endif

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
  case ModuleType::GELU_T: {
    return std::make_shared<GeLU>();
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
  case ModuleType::MIGRATE_CPU_T: {
    return std::make_shared<MigrateCpu>();
  }
  case ModuleType::MIGRATE_GPU_T: {
    return std::make_shared<MigrateGpu>();
  }
  case ModuleType::MEAN_CENTER_T: {
    symint axis;
    Serializer::read_symint(is, axis);
    return std::make_shared<MeanCenter>(axis);
  }
  case ModuleType::SOFTMAX_T: {
    symint axis;
    Serializer::read_symint(is, axis);
    return std::make_shared<Softmax>(axis);
  }
  case ModuleType::LOGSOFTMAX_T: {
    symint axis;
    Serializer::read_symint(is, axis);
    return std::make_shared<LogSoftmax>(axis);
  }
  case ModuleType::FLATTEN_T: {
    symint axis;
    Serializer::read_symint(is, axis);
    return std::make_shared<Flatten>(axis);
  }
  case ModuleType::MEAN_T: {
    symint axis;
    Serializer::read_symint(is, axis);
    return std::make_shared<Mean>(axis);
  }
  case ModuleType::MAX_T: {
    symint axis;
    Serializer::read_symint(is, axis);
    return std::make_shared<Max>(axis);
  }
  case ModuleType::MIN_T: {
    symint axis;
    Serializer::read_symint(is, axis);
    return std::make_shared<Min>(axis);
  }
  case ModuleType::VARIANCE_T: {
    symint axis;
    Serializer::read_symint(is, axis);
    return std::make_shared<Variance>(axis);
  }
  case ModuleType::STDDEV_T: {
    symint axis;
    Serializer::read_symint(is, axis);
    return std::make_shared<Stddev>(axis);
  }
  case ModuleType::RESHAPE_T: {
    tcapint sz;
    Serializer::read_tcapint(is, sz);
    std::vector<symint> shape(sz);
    for (tcapint i = 0U; i < sz; ++i) {
      Serializer::read_symint(is, shape[i]);
    }
    return std::make_shared<Reshape>(shape);
  }
  case ModuleType::MULTIHEAD_ATTENTION_T: {
    MultiHeadAttentionPtr m = std::make_shared<MultiHeadAttention>();
    Serializer::read_symint(is, m->d_model);
    Serializer::read_symint(is, m->num_heads);
    Serializer::read_symint(is, m->head_dim);
    m->W_q = std::dynamic_pointer_cast<Linear>(Linear::load(is));
    m->W_k = std::dynamic_pointer_cast<Linear>(Linear::load(is));
    m->W_v = std::dynamic_pointer_cast<Linear>(Linear::load(is));
    m->W_o = std::dynamic_pointer_cast<Linear>(Linear::load(is));

    return m;
  }
  case ModuleType::TRANSFORMER_ENCODER_LAYER_T: {
    TransformerEncoderLayerPtr t = std::make_shared<TransformerEncoderLayer>();
    Serializer::read_tcapint(is, t->d_model);
    Serializer::read_tcapint(is, t->d_ff);
    Serializer::read_tcapint(is, t->num_heads);
    t->self_attn = std::dynamic_pointer_cast<MultiHeadAttention>(
        MultiHeadAttention::load(is));
    t->ff1 = std::dynamic_pointer_cast<Linear>(Linear::load(is));
    t->ff2 = std::dynamic_pointer_cast<Linear>(Linear::load(is));
    t->norm1 = std::dynamic_pointer_cast<LayerNorm>(LayerNorm::load(is));
    t->norm2 = std::dynamic_pointer_cast<LayerNorm>(LayerNorm::load(is));
    t->activation = load(is);

    return t;
  }
  case POSITIONAL_ENCODING_T: {
    tcapint max_seq_len, d_model;
    Serializer::read_tcapint(is, max_seq_len);
    Serializer::read_tcapint(is, d_model);
    return std::make_shared<PositionalEncoding>(max_seq_len, d_model);
  }
  case LEARNED_POSITIONAL_ENCODING_T: {
    LearnedPositionalEncodingPtr l =
        std::make_shared<LearnedPositionalEncoding>();
    tcapint max_len, d_model;
    Serializer::read_tcapint(is, max_len);
    Serializer::read_tcapint(is, d_model);
    l->max_len = max_len;
    l->d_model = d_model;
    l->pos_encoding = Parameter::load(is);

    return l;
  }
#if QRACK_AVAILABLE
  case QRACK_NEURON_LAYER_T: {
    tcapint input_q, output_q, hidden_q;
    Serializer::read_tcapint(is, input_q);
    Serializer::read_tcapint(is, output_q);
    Serializer::read_tcapint(is, hidden_q);
    tcapint lowest_combo, highest_combo;
    Serializer::read_tcapint(is, lowest_combo);
    Serializer::read_tcapint(is, highest_combo);
    QuantumFunctionType pre_qfn;
    Serializer::read_quantum_fn(is, pre_qfn);
    QuantumFunctionType post_qfn;
    Serializer::read_quantum_fn(is, post_qfn);
    Qrack::QNeuronActivationFn activation_fn;
    Serializer::read_qneuron_activation_fn(is, activation_fn);
    tcapint mask;
    Serializer::read_tcapint(is, mask);
    const bool md = (mask & 1U);
    const bool sd = (mask & 2U);
    const bool bdt = (mask & 4U);
    const bool tn = (mask & 8U);
    const bool hp = (mask & 16U);
    const bool sp = (mask & 32U);

    QrackNeuronLayerPtr qnl = std::make_shared<QrackNeuronLayer>(
        input_q, output_q, hidden_q, lowest_combo, highest_combo, pre_qfn,
        post_qfn, activation_fn, nullptr, nullptr, md, sd, bdt, tn, hp, sp);

    for (size_t i = 0U; i < qnl->neurons.size(); ++i) {
      const auto &n = qnl->neurons[i];
      n->angles = Parameter::load(is);
      n->data =
          static_cast<CpuRealStorage *>(n->angles->storage.get())->data.get();
    }

    qnl->update_param_vector();

    return qnl;
  }
#endif
  case ModuleType::NONE_MODULE_TYPE:
  default:
    throw std::domain_error("Can't recognize ModuleType in Module::load!");
  }

  return nullptr;
}
} // namespace Weed
