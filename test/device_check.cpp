//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of
// Qrack.
//
// This file was produced almost in its entirety, verbatim, by (Anthropic)
// Claude.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

// Differential check of the GPU kernels against the CPU kernels.
//
// Each operation is run, with its gradient, over shapes that stress kernel
// indexing: a leading index of extent 1, partial broadcasts, and transposed
// and sliced views, for real and for complex tensors. Tensors are placed on a
// device by their size, so the reference and the device run have to be
// separate processes, with WEED_GSTRIDE=1 to send everything to the device:
//
//     ./device_check cpu ref.txt
//     ./device_check cpu ref_c.txt complex
//     WEED_GSTRIDE=1 ./device_check gpu dev.txt
//     WEED_GSTRIDE=1 ./device_check gpu dev_c.txt complex
//     ./device_check compare ref.txt dev.txt
//     ./device_check compare ref_c.txt dev_c.txt

#include "modules/embedding.hpp"
#include "modules/layernorm.hpp"
#include "modules/linear.hpp"
#include "ops/triu_fill.hpp"
#include "tensors/complex_tensor.hpp"
#include "tensors/real_tensor.hpp"
#include "tensors/symbol_tensor.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
using namespace Weed;
typedef std::vector<tcapint> Shape;
static bool g_complex = false;
static std::vector<double> vec(TensorPtr t) {
  TensorPtr c = Tensor::contiguous(Tensor::detach(t)->cast(DeviceTag::CPU));
  if (c->storage->dtype == DType::COMPLEX) {
    ComplexTensor *r = static_cast<ComplexTensor *>(c.get());
    std::vector<double> v(2 * c->get_broadcast_size());
    for (size_t i = 0; i < v.size() / 2; ++i) {
      const complex z = (*r)[i];
      v[2 * i] = z.real();
      v[2 * i + 1] = z.imag();
    }
    return v;
  }
  RealTensor *r = static_cast<RealTensor *>(c.get());
  std::vector<double> v(c->get_broadcast_size());
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = (*r)[i];
  return v;
}
static TensorPtr mkc(const Shape &s, double seed, bool rg, DeviceTag d) {
  tcapint n = 1;
  for (auto x : s)
    n *= x;
  std::vector<complex> v(n);
  for (tcapint i = 0; i < n; ++i)
    v[i] = complex((real1)std::sin(seed + 1.37 * i),
                   (real1)std::cos(0.7 * seed + 0.91 * i));
  return std::make_shared<Tensor>(v, s, rg, d);
}
static TensorPtr mk(const Shape &s, double seed, bool rg, DeviceTag d,
                    double lo = -1.0) {
  tcapint n = 1;
  for (auto x : s)
    n *= x;
  std::vector<real1> v(n);
  for (tcapint i = 0; i < n; ++i)
    v[i] = (real1)(lo < 0 ? std::sin(seed + 1.37 * i)
                          : (lo + 0.5 + 0.5 * std::sin(seed + 1.37 * i)));
  return std::make_shared<Tensor>(v, s, rg, d);
}
struct Res {
  std::vector<double> y, gx, gp;
  std::vector<tcapint> shape;
  std::string err;
};
typedef std::function<TensorPtr(TensorPtr, TensorPtr, DeviceTag)> Fn;
static Res run(const Shape &xs, const Shape &ps, Fn f, DeviceTag d, bool grad,
               double lo) {
  Res r;
  try {
    TensorPtr x = g_complex ? mkc(xs, 0.3, grad, d) : mk(xs, 0.3, grad, d, lo);
    TensorPtr p = ps.empty() ? nullptr : mk(ps, 1.9, grad, d, lo);
    TensorPtr y = f(x, p, d);
    r.y = vec(y);
    r.shape = y->shape;
    if (grad) {
      const size_t wn = y->get_broadcast_size();
      std::vector<real1> w(wn);
      for (size_t i = 0; i < w.size(); ++i)
        w[i] = (real1)std::cos(0.4 + 0.77 * i);
      // weights built on CPU-contiguous layout of y, moved to the device
      TensorPtr wt = std::make_shared<Tensor>(w, y->shape, false, d);
      Tensor::backward(Tensor::sum(Tensor::contiguous(y) * wt));
      if (x->grad)
        r.gx = vec(x->grad);
      if (p && p->grad) {
        p->reset_broadcast();
        r.gp = vec(p->grad);
      }
    }
  } catch (const std::exception &e) {
    r.err = e.what();
  }
  return r;
}
static FILE *outf = nullptr;
static DeviceTag g_dev = DeviceTag::CPU;
static void dumpv(const char *tag, const std::vector<double> &v) {
  fprintf(outf, " %s %zu", tag, v.size());
  for (double x : v)
    fprintf(outf, " %.9g", x);
}
static void check(const std::string &name, const Shape &xs, const Shape &ps,
                  Fn f, bool grad = true, double lo = -1.0) {
  Res r = run(xs, ps, f, g_dev, grad, lo);
  std::string sh = "(";
  for (auto s : xs)
    sh += std::to_string(s) + ",";
  sh += ")";
  if (!ps.empty()) {
    sh += "(";
    for (auto s : ps)
      sh += std::to_string(s) + ",";
    sh += ")";
  }
  std::string key = name + sh;
  for (auto &c : key)
    if (c == ' ')
      c = '_';
  fprintf(outf, "%s", key.c_str());
  if (!r.err.empty()) {
    std::string e = r.err;
    for (auto &c : e)
      if (c == ' ')
        c = '_';
    fprintf(outf, " ERR %s\n", e.c_str());
    fflush(outf);
    return;
  }
  std::vector<double> shp(r.shape.begin(), r.shape.end());
  dumpv("shape", shp);
  dumpv("y", r.y);
  dumpv("gx", r.gx);
  dumpv("gp", r.gp);
  fprintf(outf, "\n");
  fflush(outf);
}

// Compare two result files: each line is a check name, then either "ERR" and a
// message, or a list of named vectors ("<tag> <count> <values...>").
struct Record {
  bool is_error = false;
  std::string message;
  std::map<std::string, std::vector<double>> vectors;
};
static std::vector<std::pair<std::string, Record>>
load_records(const char *path) {
  std::vector<std::pair<std::string, Record>> records;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string key, tag;
    if (!(ls >> key)) {
      continue;
    }
    Record r;
    while (ls >> tag) {
      if (tag == "ERR") {
        r.is_error = true;
        ls >> r.message;
        break;
      }
      size_t n;
      ls >> n;
      std::vector<double> v(n);
      for (size_t k = 0U; k < n; ++k) {
        ls >> v[k];
      }
      r.vectors[tag] = v;
    }
    records.emplace_back(key, r);
  }

  return records;
}
static int compare(const char *ref_path, const char *dev_path) {
  const auto ref = load_records(ref_path);
  std::map<std::string, Record> dev;
  for (const auto &kv : load_records(dev_path)) {
    dev[kv.first] = kv.second;
  }

  size_t n_ok = 0U, n_bad = 0U, n_ref_error = 0U;
  for (const auto &kv : ref) {
    const std::string &key = kv.first;
    const Record &r = kv.second;
    if (r.is_error) {
      // Not a device difference: the reference itself can't do this.
      ++n_ref_error;
      continue;
    }
    std::string why;
    const auto it = dev.find(key);
    if (it == dev.end()) {
      why = "no result (did the run crash?)";
    } else if (it->second.is_error) {
      why = "threw: " + it->second.message;
    } else {
      for (const auto &tv : r.vectors) {
        const std::vector<double> &a = tv.second;
        const auto jt = it->second.vectors.find(tv.first);
        if ((jt == it->second.vectors.end()) ||
            (jt->second.size() != a.size())) {
          why = tv.first + " has a different size";
          break;
        }
        double worst = 0.0;
        for (size_t k = 0U; k < a.size(); ++k) {
          const double tol = 2e-4 + 2e-3 * std::abs(a[k]);
          const double d = std::abs(a[k] - jt->second[k]);
          if (!(d <= tol)) {
            worst = std::max(worst, std::isfinite(d) ? d : 1e30);
          }
        }
        if (worst > 0.0) {
          why = tv.first + " differs by " + std::to_string(worst);
          break;
        }
      }
    }
    if (why.empty()) {
      ++n_ok;
    } else {
      ++n_bad;
      std::cout << "DIFFERS  " << key << ": " << why << std::endl;
    }
  }
  std::cout << n_ok << "/" << (n_ok + n_bad) << " checks agree";
  if (n_ref_error) {
    std::cout << " (" << n_ref_error << " more threw in the reference run)";
  }
  std::cout << std::endl;

  return n_bad ? 1 : 0;
}

int main(int argc, char **argv) {
  if ((argc < 3) || ((std::string(argv[1]) == "compare") && (argc < 4))) {
    std::cout << "usage: device_check cpu|gpu <out file> [complex]\n"
              << "       device_check compare <reference file> <device file>"
              << std::endl;
    return 2;
  }
  if (std::string(argv[1]) == "compare") {
    return compare(argv[2], argv[3]);
  }
  g_dev = (std::string(argv[1]) == "gpu") ? DeviceTag::GPU : DeviceTag::CPU;
  outf = fopen(argv[2], "w");
  if (argc > 3) {
    // complex input (and real parameters, for the mixed-type kernels)
    g_complex = true;
    const std::vector<Shape> cshapes{
        {6}, {2, 3}, {1, 6}, {2, 3, 4}, {1, 3, 4}, {2, 1, 4}, {1, 2, 3, 4}};
    for (const Shape &s : cshapes) {
      check("c:x*x", s, {},
            [](TensorPtr x, TensorPtr, DeviceTag) { return x * x; });
      check("c:x+x*x", s, {},
            [](TensorPtr x, TensorPtr, DeviceTag) { return x + x * x; });
      check("c:x-0.5x", s, {}, [](TensorPtr x, TensorPtr, DeviceTag) {
        return x - x * real1(0.5f);
      });
      check("c:x/(x*x+2)", s, {}, [](TensorPtr x, TensorPtr, DeviceTag) {
        return x / (x * x + real1(2));
      });
      check("c:abs", s, {},
            [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::abs(x); });
      check("c:pow(x,2)", s, {},
            [](TensorPtr x, TensorPtr, DeviceTag) { return x ^ real1(2); });
      check("c:exp", s, {},
            [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::exp(x); });
      check("c:sum()", s, {},
            [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::sum(x); });
      check("c:mean()", s, {},
            [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::mean(x); });
      for (symint ax = 0; ax < (symint)s.size(); ++ax) {
        const std::string a = std::to_string(ax);
        check("c:sum(axis " + a + ")", s, {},
              [ax](TensorPtr x, TensorPtr, DeviceTag) {
                return Tensor::sum(x * real1(1), ax);
              });
        check("c:mean(axis " + a + ")", s, {},
              [ax](TensorPtr x, TensorPtr, DeviceTag) {
                return Tensor::mean(x * real1(1), ax);
              });
        if (s[ax] > 1)
          check("c:slice(axis " + a + ")", s, {},
                [ax](TensorPtr x, TensorPtr, DeviceTag) {
                  return Tensor::slice(x * real1(1), ax, 1, x->shape[ax] - 1) *
                         real1(2);
                });
      }
      if (s.size() >= 2) {
        check("c:transpose(0,-1)*2", s, {},
              [](TensorPtr x, TensorPtr, DeviceTag) {
                return Tensor::transpose(x, 0, -1) * real1(2);
              });
        check("c:x @ x^T", s, {}, [](TensorPtr x, TensorPtr, DeviceTag) {
          return x >> Tensor::transpose(x, -2, -1);
        });
        check("c:x @ W(real)", s, Shape{s.back(), 3},
              [](TensorPtr x, TensorPtr p, DeviceTag) { return x >> p; });
        check("c:W(real)^T-ish @ x", s, Shape{3, s[s.size() - 2]},
              [](TensorPtr x, TensorPtr p, DeviceTag) {
                return (x->shape.size() == 2U) ? (p >> x) : (x * real1(1));
              });
      }
      const Shape ps{s.back()};
      check("c:x + bias(real)", s, ps,
            [](TensorPtr x, TensorPtr p, DeviceTag) { return x + p; });
      check("c:x * gamma(real)", s, ps,
            [](TensorPtr x, TensorPtr p, DeviceTag) { return x * p; });
      check("c:x - bias(real)", s, ps,
            [](TensorPtr x, TensorPtr p, DeviceTag) { return x - p; });
      check("c:bias(real) - x", s, ps,
            [](TensorPtr x, TensorPtr p, DeviceTag) { return p - x; });
      check("c:x / (gamma^2+1)", s, ps,
            [](TensorPtr x, TensorPtr p, DeviceTag) {
              return x / (p * p + real1(1));
            });
      check("c:gamma / (x*x+2)", s, ps,
            [](TensorPtr x, TensorPtr p, DeviceTag) {
              return p / (x * x + real1(2));
            });
    }
    fclose(outf);
    return 0;
  }
  const std::vector<Shape> shapes{{6},       {2, 3},    {1, 6},
                                  {6, 1},    {2, 3, 4}, {1, 3, 4},
                                  {2, 1, 4}, {1, 1, 5}, {1, 2, 3, 4}};
  for (const Shape &s : shapes) {
    check("x*x", s, {},
          [](TensorPtr x, TensorPtr, DeviceTag) { return x * x; });
    check("x+x*x", s, {},
          [](TensorPtr x, TensorPtr, DeviceTag) { return x + x * x; });
    check("x-0.5x", s, {}, [](TensorPtr x, TensorPtr, DeviceTag) {
      return x - x * real1(0.5f);
    });
    check("x/(x*x+1)", s, {}, [](TensorPtr x, TensorPtr, DeviceTag) {
      return x / (x * x + real1(1));
    });
    check("relu", s, {},
          [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::relu(x); });
    check("sigmoid", s, {},
          [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::sigmoid(x); });
    check("tanh", s, {},
          [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::tanh(x); });
    check("gelu", s, {},
          [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::gelu(x); });
    check("sin", s, {},
          [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::sin(x); });
    check("cos", s, {},
          [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::cos(x); });
    check("abs", s, {},
          [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::abs(x); });
    check("clamp", s, {}, [](TensorPtr x, TensorPtr, DeviceTag) {
      return Tensor::clamp(x, real1(-0.3f), real1(0.4f));
    });
    check(
        "pow(x,2)", s, {},
        [](TensorPtr x, TensorPtr, DeviceTag) { return x ^ real1(2); }, true,
        0.5);
    check(
        "pow(x,0.5)", s, {},
        [](TensorPtr x, TensorPtr, DeviceTag) { return x ^ real1(0.5f); }, true,
        0.5);
    check("exp", s, {},
          [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::exp(x); });
    check(
        "log", s, {},
        [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::log(x); }, true,
        0.5);
    check("sum()", s, {},
          [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::sum(x); });
    check("mean()", s, {},
          [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::mean(x); });
    check("max()", s, {},
          [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::max(x); });
    check("min()", s, {},
          [](TensorPtr x, TensorPtr, DeviceTag) { return Tensor::min(x); });
    for (symint ax = 0; ax < (symint)s.size(); ++ax) {
      const std::string a = std::to_string(ax);
      check("sum(axis " + a + ")", s, {},
            [ax](TensorPtr x, TensorPtr, DeviceTag) {
              return Tensor::sum(x * real1(1), ax);
            });
      check("mean(axis " + a + ")", s, {},
            [ax](TensorPtr x, TensorPtr, DeviceTag) {
              return Tensor::mean(x * real1(1), ax);
            });
      check("max(axis " + a + ")", s, {},
            [ax](TensorPtr x, TensorPtr, DeviceTag) {
              return Tensor::max(x * real1(1), ax);
            });
      check("min(axis " + a + ")", s, {},
            [ax](TensorPtr x, TensorPtr, DeviceTag) {
              return Tensor::min(x * real1(1), ax);
            });
      check("softmax(axis " + a + ")", s, {},
            [ax](TensorPtr x, TensorPtr, DeviceTag) {
              return Tensor::softmax(x, ax);
            });
      check("logsoftmax(axis " + a + ")", s, {},
            [ax](TensorPtr x, TensorPtr, DeviceTag) {
              return Tensor::logsoftmax(x, ax);
            });
      if (s[ax] > 1)
        check("slice(axis " + a + ")", s, {},
              [ax](TensorPtr x, TensorPtr, DeviceTag) {
                return Tensor::slice(x * real1(1), ax, 1, x->shape[ax] - 1) *
                       real1(2);
              });
    }
    if (s.size() >= 2) {
      check("transpose(0,-1)*2", s, {}, [](TensorPtr x, TensorPtr, DeviceTag) {
        return Tensor::transpose(x, 0, -1) * real1(2);
      });
      check("transpose + x^T", s, {}, [](TensorPtr x, TensorPtr, DeviceTag) {
        TensorPtr t = Tensor::transpose(x, -2, -1);
        return t + t * t;
      });
      check("x @ x^T", s, {}, [](TensorPtr x, TensorPtr, DeviceTag) {
        return x >> Tensor::transpose(x, -2, -1);
      });
      check("reshape(-1) of transpose", s, {},
            [](TensorPtr x, TensorPtr, DeviceTag) {
              return Tensor::reshape(Tensor::transpose(x, 0, -1), {-1}) *
                     real1(3);
            });
    }
    // broadcast against a parameter over the last index (bias / gamma)
    const Shape ps{s.back()};
    check("x + bias", s, ps,
          [](TensorPtr x, TensorPtr p, DeviceTag) { return x + p; });
    check("x * gamma", s, ps,
          [](TensorPtr x, TensorPtr p, DeviceTag) { return x * p; });
    check("x - bias", s, ps,
          [](TensorPtr x, TensorPtr p, DeviceTag) { return x - p; });
    check("x / (gamma^2+1)", s, ps, [](TensorPtr x, TensorPtr p, DeviceTag) {
      return x / (p * p + real1(1));
    });
    check("x - mean(x,-1)", s, {}, [](TensorPtr x, TensorPtr, DeviceTag) {
      return x - Tensor::mean(x, -1);
    });
    check("layernorm core", s, {}, [](TensorPtr x, TensorPtr, DeviceTag) {
      TensorPtr xc = x - Tensor::mean(x, -1);
      return xc / ((Tensor::mean(xc * xc, -1) + real1(1e-5f)) ^ real1(0.5f));
    });
    if (s.size() >= 2) {
      const Shape ws{s.back(), 3};
      check("x @ W", s, ws,
            [](TensorPtr x, TensorPtr p, DeviceTag) { return x >> p; });
    }
  }
  // triu_fill
  for (const Shape &s : std::vector<Shape>{{3, 3}, {2, 5}, {4, 2}, {1, 4}}) {
    for (tcapint dg : {1U, 2U, 3U}) {
      check(
          "triu_fill d=" + std::to_string(dg), s, {},
          [dg](TensorPtr x, TensorPtr, DeviceTag d) {
            TensorPtr m = Tensor::zeros(x->shape, false, false, DType::REAL, d);
            Weed::triu_fill(*m, real1(-7), dg);
            return x + m;
          },
          false);
    }
  }
  // embedding: (T), (1, T), (B, T)
  for (const Shape &s :
       std::vector<Shape>{{5}, {1}, {1, 5}, {2, 3}, {3, 1}, {1, 1}}) {
    std::string idx_shape = "(";
    for (auto e : s) {
      idx_shape += std::to_string(e) + ",";
    }
    idx_shape += ")";
    check(
        "embedding of indices " + idx_shape, {4, 3}, {},
        [s](TensorPtr w, TensorPtr, DeviceTag d) {
          tcapint n = 1;
          for (auto x : s)
            n *= x;
          std::vector<symint> ids(n);
          for (tcapint i = 0; i < n; ++i)
            ids[i] = (symint)((i * 7 + 1) % 4);
          EmbeddingPtr e = std::make_shared<Embedding>(4, 3, DType::REAL, d);
          e->weight = std::make_shared<Parameter>(
              std::vector<real1>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
              Shape{4, 3}, d);
          if (!w->requires_grad)
            e->eval();
          TensorPtr y =
              e->forward(std::make_shared<SymbolTensor>(ids, s, false, d));
          return y * Tensor::reshape(Tensor::slice(w, 0, 0, 1), {3});
        },
        true);
  }
  fclose(outf);
  return 0;
}
