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

#include "modules/module.hpp"

namespace Weed {
/**
 * Standard interface for sequential models of multiple layers
 */
struct Sequential : public Module {
  std::vector<ModulePtr> layers;
  std::vector<ParameterPtr> param_vector;
  // Sequence offset set by a containing module (0 at top level)
  tcapint seq_offset;

  Sequential(const std::vector<ModulePtr> &l)
      : Module(SEQUENTIAL_T), layers(l), param_vector(), seq_offset(0U) {
    for (size_t i = 0U; i < layers.size(); ++i) {
      const std::vector<ParameterPtr> p = layers[i]->parameters();
      param_vector.insert(param_vector.end(), p.begin(), p.end());
    }
  }

  void train() override {
    for (const ModulePtr &m : layers) {
      m->train();
    }
  }
  void eval() override {
    for (const ModulePtr &m : layers) {
      m->eval();
    }
  }

  void migrate_cpu() override {
    for (const ModulePtr &m : layers) {
      m->migrate_cpu();
    }
  }
  void migrate_gpu() override {
    for (const ModulePtr &m : layers) {
      m->migrate_gpu();
    }
  }

  void set_max_kv_seq_len(tcapint m) override {
    for (const ModulePtr &md : layers) {
      md->set_max_kv_seq_len(m);
    }
  }

  void reset_cache() override {
    seq_offset = 0U;
    for (const ModulePtr &m : layers) {
      m->reset_cache();
    }
  }

  tcapint get_kv_cache_len() const override {
    tcapint len = 0U;
    for (const ModulePtr &m : layers) {
      len = std::max(len, m->get_kv_cache_len());
    }

    return len;
  }

  void set_seq_offset(tcapint o) override {
    seq_offset = o;
    for (const ModulePtr &m : layers) {
      m->set_seq_offset(o);
    }
  }

  /**
   * Tell positional encodings how many tokens are already in the KV cache,
   * since (with a KV cache) forward() is only passed the new tokens.
   */
  void sync_seq_offset() {
    const tcapint o = std::max(seq_offset, get_kv_cache_len());
    for (const ModulePtr &m : layers) {
      m->set_seq_offset(o);
    }
  }

  TensorPtr forward(const TensorPtr x) override {
    sync_seq_offset();
    TensorPtr tmp = x;
    for (size_t i = 0U; i < layers.size(); ++i) {
      tmp = layers[i]->forward(tmp);
    }

    return tmp;
  }
  TensorPtr forward(const SymbolTensorPtr x) override {
    if (layers.empty()) {
      return std::make_shared<Tensor>();
    }
    sync_seq_offset();
    TensorPtr tmp = layers[0]->forward(x);
    for (size_t i = 1U; i < layers.size(); ++i) {
      tmp = layers[i]->forward(tmp);
    }

    return tmp;
  }

  std::vector<ParameterPtr> parameters() override { return param_vector; }

  void save(std::ostream &) const override;
};
typedef std::shared_ptr<Sequential> SequentialPtr;
} // namespace Weed
