#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace py = pybind11;

namespace {

inline int64_t lut_size_for_n(int n) {
  TORCH_CHECK(n >= 1 && n < 31, "n must be in [1, 30]");
  return int64_t{1} << n;
}

inline int compact_addr_without_bit(uint32_t addr, int bit) {
  const uint32_t low_mask = (uint32_t{1} << bit) - 1u;
  const uint32_t low = addr & low_mask;
  const uint32_t high = addr >> (bit + 1);
  return static_cast<int>(low | (high << bit));
}

inline uint32_t expand_addr(uint32_t compact, int bit) {
  const uint32_t low_mask = (uint32_t{1} << bit) - 1u;
  const uint32_t low = compact & low_mask;
  const uint32_t high = compact >> bit;
  return low | (high << (bit + 1));
}

inline void make_efd_derivatives_runtime(
    const float* lut, float* deriv, int fan_in, float alpha, float beta) {
  const int half = 1 << (fan_in - 1);
  for (int bit = 0; bit < fan_in; ++bit) {
    float* v = deriv + static_cast<int64_t>(bit) * half;
    const uint32_t full_bit = uint32_t{1} << bit;
    for (int q = 0; q < half; ++q) {
      const uint32_t base = expand_addr(static_cast<uint32_t>(q), bit);
      v[q] = lut[base | full_bit] - lut[base];
    }
    for (int dim = 0; dim < fan_in - 1; ++dim) {
      const int step = 1 << dim;
      const int block = step << 1;
      for (int start = 0; start < half; start += block) {
        for (int i = 0; i < step; ++i) {
          const int ai = start + i;
          const int bi = ai + step;
          const float a = v[ai];
          const float b = v[bi];
          v[ai] = a + beta * b;
          v[bi] = beta * a + b;
        }
      }
    }
    for (int q = 0; q < half; ++q) v[q] *= alpha;
  }
}

template <int N>
inline void make_efd_derivatives_fixed(
    const float* lut, float* deriv, float alpha, float beta) {
  constexpr int half = 1 << (N - 1);
  for (int bit = 0; bit < N; ++bit) {
    float* v = deriv + bit * half;
    const uint32_t full_bit = uint32_t{1} << bit;
    for (int q = 0; q < half; ++q) {
      const uint32_t base = expand_addr(static_cast<uint32_t>(q), bit);
      v[q] = lut[base | full_bit] - lut[base];
    }
    for (int dim = 0; dim < N - 1; ++dim) {
      const int step = 1 << dim;
      const int block = step << 1;
      for (int start = 0; start < half; start += block) {
        for (int i = 0; i < step; ++i) {
          const int ai = start + i;
          const int bi = ai + step;
          const float a = v[ai];
          const float b = v[bi];
          v[ai] = a + beta * b;
          v[bi] = beta * a + b;
        }
      }
    }
    for (int q = 0; q < half; ++q) v[q] *= alpha;
  }
}

struct LayerState {
  int64_t input_size = 0;
  int num_luts = 0;
  int fan_in = 0;
  int64_t lut_size = 0;
  std::vector<float> luts;
  std::vector<int32_t> mapping;

  LayerState() = default;
  LayerState(int64_t in, int nl, int n)
      : input_size(in), num_luts(nl), fan_in(n), lut_size(lut_size_for_n(n)),
        luts(static_cast<size_t>(static_cast<int64_t>(nl) * lut_size)),
        mapping(static_cast<size_t>(static_cast<int64_t>(nl) * n)) {}
};

struct FirstLearnableMapping {
  int64_t input_size = 0;
  int num_luts = 0;
  int fan_in = 0;
  int64_t num_slots = 0;
  std::vector<float> weights;

  FirstLearnableMapping(int64_t in, int nl, int n)
      : input_size(in), num_luts(nl), fan_in(n),
        num_slots(static_cast<int64_t>(nl) * n),
        weights(static_cast<size_t>(num_slots * input_size)) {}

  inline float* slot(int64_t s) { return weights.data() + s * input_size; }
  inline const float* slot(int64_t s) const { return weights.data() + s * input_size; }
};

void check_common(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const std::vector<torch::Tensor>& initial_luts,
    int fan_in,
    float tau,
    float learning_rate,
    int64_t epochs,
    int64_t batch_size,
    int64_t lr_decay_steps,
    float gamma,
    int64_t num_classes) {
  TORCH_CHECK(x.device().is_cpu() && y.device().is_cpu(), "x and y must be on CPU");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "x and y must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kUInt8, "x must be torch.uint8");
  TORCH_CHECK(y.scalar_type() == torch::kInt64, "y must be torch.int64");
  TORCH_CHECK(x.dim() == 2 && x.size(0) > 0 && x.size(1) > 0,
              "x must be non-empty [num_samples, input_size]");
  TORCH_CHECK(y.dim() == 1 && y.size(0) == x.size(0), "y must have shape [num_samples]");
  TORCH_CHECK(initial_luts.size() >= 2, "multilayer kernel requires at least 2 layers");
  TORCH_CHECK(num_classes >= 2, "num_classes must be >= 2");
  TORCH_CHECK(tau > 0.0f && std::isfinite(tau), "tau must be finite and > 0");
  TORCH_CHECK(learning_rate > 0.0f && std::isfinite(learning_rate),
              "learning_rate must be finite and > 0");
  TORCH_CHECK(epochs >= 0, "epochs must be >= 0");
  TORCH_CHECK(batch_size > 0, "batch_size must be > 0");
  TORCH_CHECK(lr_decay_steps > 0, "lr_decay_steps must be > 0");
  TORCH_CHECK(gamma > 0.0f && std::isfinite(gamma), "gamma must be finite and > 0");
  const int64_t lut_size = lut_size_for_n(fan_in);
  for (size_t l = 0; l < initial_luts.size(); ++l) {
    const auto& t = initial_luts[l];
    TORCH_CHECK(t.device().is_cpu() && t.is_contiguous() &&
                    t.scalar_type() == torch::kFloat32 && t.dim() == 2,
                "each initial_luts tensor must be contiguous CPU float32 [num_luts, 2^n]");
    TORCH_CHECK(t.size(0) > 0 && t.size(1) == lut_size,
                "initial_luts has wrong LUT size for n");
  }
  TORCH_CHECK(initial_luts.back().size(0) % num_classes == 0,
              "last layer num_luts must be divisible by num_classes");
  const int64_t* yp = y.data_ptr<int64_t>();
  for (int64_t i = 0; i < y.size(0); ++i) {
    TORCH_CHECK(yp[i] >= 0 && yp[i] < num_classes,
                "y contains a class outside [0, num_classes)");
  }
}

void check_mapping_tensor(
    const torch::Tensor& m, int64_t num_luts, int fan_in, int64_t input_size) {
  TORCH_CHECK(m.device().is_cpu() && m.is_contiguous() &&
                  m.scalar_type() == torch::kInt32 && m.dim() == 2,
              "mapping tensors must be contiguous CPU int32");
  TORCH_CHECK(m.size(0) == num_luts && m.size(1) == fan_in,
              "mapping has wrong shape");
  const int32_t* p = m.data_ptr<int32_t>();
  for (int64_t i = 0; i < num_luts * fan_in; ++i) {
    TORCH_CHECK(p[i] >= 0 && p[i] < input_size,
                "mapping contains an out-of-range index");
  }
}

void resolve_first_mapping(FirstLearnableMapping& lm, std::vector<int32_t>& mapping) {
  at::parallel_for(0, lm.num_slots, 1, [&](int64_t begin, int64_t end) {
    for (int64_t s = begin; s < end; ++s) {
      const float* w = lm.slot(s);
      int32_t best = 0;
      float best_v = w[0];
      for (int64_t f = 1; f < lm.input_size; ++f) {
        if (w[f] > best_v) {
          best_v = w[f];
          best = static_cast<int32_t>(f);
        }
      }
      mapping[static_cast<size_t>(s)] = best;
    }
  });
}

template <int FixedN>
inline void make_layer_derivatives(
    const LayerState& layer, std::vector<float>& deriv_all, float alpha, float beta) {
  const int half = 1 << (layer.fan_in - 1);
  const int64_t per_lut = static_cast<int64_t>(layer.fan_in) * half;
  deriv_all.resize(static_cast<size_t>(static_cast<int64_t>(layer.num_luts) * per_lut));
  at::parallel_for(0, layer.num_luts, 1, [&](int64_t begin, int64_t end) {
    for (int64_t j = begin; j < end; ++j) {
      const float* lut = layer.luts.data() + j * layer.lut_size;
      float* d = deriv_all.data() + j * per_lut;
      if constexpr (FixedN > 0) {
        make_efd_derivatives_fixed<FixedN>(lut, d, alpha, beta);
      } else {
        make_efd_derivatives_runtime(lut, d, layer.fan_in, alpha, beta);
      }
    }
  });
}

void clamp_layer(LayerState& layer) {
  at::parallel_for(0, static_cast<int64_t>(layer.luts.size()), 256,
                   [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      layer.luts[static_cast<size_t>(i)] =
          std::max(-1.0f, std::min(1.0f, layer.luts[static_cast<size_t>(i)]));
    }
  });
}

template <int FixedN>
std::vector<torch::Tensor> train_multi_impl(
    torch::Tensor x,
    torch::Tensor y,
    std::vector<torch::Tensor> initial_luts,
    torch::Tensor first_mapping_weights,
    std::vector<torch::Tensor> fixed_mappings,
    int runtime_n,
    float tau,
    float learning_rate,
    int64_t epochs,
    int64_t batch_size,
    int64_t lr_decay_steps,
    float gamma,
    int64_t num_classes64,
    bool first_learnable,
    bool verbose) {
  const int fan_in = FixedN > 0 ? FixedN : runtime_n;
  check_common(x, y, initial_luts, fan_in, tau, learning_rate, epochs,
               batch_size, lr_decay_steps, gamma, num_classes64);

  const int L = static_cast<int>(initial_luts.size());
  const int num_classes = static_cast<int>(num_classes64);
  const int64_t input_size0 = x.size(1);
  const int64_t n_samples = x.size(0);
  const int64_t lut_size = lut_size_for_n(fan_in);
  const int half = 1 << (fan_in - 1);
  const float alpha = 0.5f * std::pow(0.75f, static_cast<float>(fan_in - 1));
  constexpr float beta = 1.0f / 3.0f;

  std::vector<LayerState> layers;
  layers.reserve(static_cast<size_t>(L));
  int64_t prev_size = input_size0;
  for (int l = 0; l < L; ++l) {
    const int nl = static_cast<int>(initial_luts[static_cast<size_t>(l)].size(0));
    layers.emplace_back(prev_size, nl, fan_in);
    std::memcpy(layers.back().luts.data(), initial_luts[static_cast<size_t>(l)].data_ptr<float>(),
                static_cast<size_t>(static_cast<int64_t>(nl) * lut_size) * sizeof(float));
    prev_size = nl;
  }

  FirstLearnableMapping lm(input_size0, layers[0].num_luts, fan_in);
  if (first_learnable) {
    TORCH_CHECK(first_mapping_weights.device().is_cpu() && first_mapping_weights.is_contiguous() &&
                    first_mapping_weights.scalar_type() == torch::kFloat32 &&
                    first_mapping_weights.dim() == 2 &&
                    first_mapping_weights.size(0) == input_size0 &&
                    first_mapping_weights.size(1) == static_cast<int64_t>(layers[0].num_luts) * fan_in,
                "first_mapping_weights has wrong shape");
    const float* src = first_mapping_weights.data_ptr<float>();
    for (int64_t s = 0; s < lm.num_slots; ++s) {
      float* dst = lm.slot(s);
      for (int64_t f = 0; f < input_size0; ++f) dst[f] = src[f * lm.num_slots + s];
    }
    TORCH_CHECK(static_cast<int>(fixed_mappings.size()) == L - 1,
                "learnable first layer expects one fixed mapping for each later layer");
    for (int l = 1; l < L; ++l) {
      const auto& mt = fixed_mappings[static_cast<size_t>(l - 1)];
      check_mapping_tensor(mt, layers[l].num_luts, fan_in, layers[l].input_size);
      std::memcpy(layers[l].mapping.data(), mt.data_ptr<int32_t>(),
                  layers[l].mapping.size() * sizeof(int32_t));
    }
  } else {
    TORCH_CHECK(static_cast<int>(fixed_mappings.size()) == L,
                "fixed first layer expects one fixed mapping for every layer");
    for (int l = 0; l < L; ++l) {
      const auto& mt = fixed_mappings[static_cast<size_t>(l)];
      check_mapping_tensor(mt, layers[l].num_luts, fan_in, layers[l].input_size);
      std::memcpy(layers[l].mapping.data(), mt.data_ptr<int32_t>(),
                  layers[l].mapping.size() * sizeof(int32_t));
    }
  }

  const uint8_t* x_ptr = x.data_ptr<uint8_t>();
  const int64_t* y_ptr = y.data_ptr<int64_t>();
  const int last_n = layers.back().num_luts;
  const int luts_per_class = last_n / num_classes;

  std::vector<float> exp_neg_diff(static_cast<size_t>(luts_per_class + 1));
  for (int d = 0; d <= luts_per_class; ++d)
    exp_neg_diff[static_cast<size_t>(d)] = std::exp(-static_cast<float>(d) / tau);

  std::vector<uint8_t> x_batch(static_cast<size_t>(batch_size * input_size0));
  std::vector<int64_t> batch_y(static_cast<size_t>(batch_size));
  std::vector<int64_t> ones_offsets(static_cast<size_t>(batch_size + 1));
  std::vector<int32_t> ones_indices;
  ones_indices.reserve(static_cast<size_t>(batch_size * std::min<int64_t>(input_size0, 2048) / 2));

  std::vector<std::vector<uint8_t>> activations(static_cast<size_t>(L));
  std::vector<std::vector<uint32_t>> addresses(static_cast<size_t>(L));
  for (int l = 0; l < L; ++l) {
    activations[static_cast<size_t>(l)].resize(
        static_cast<size_t>(batch_size * layers[l].num_luts));
    addresses[static_cast<size_t>(l)].resize(
        static_cast<size_t>(batch_size * layers[l].num_luts));
  }

  std::vector<float> class_grad(static_cast<size_t>(batch_size * num_classes));
  std::vector<float> sample_loss(static_cast<size_t>(batch_size));
  std::vector<uint8_t> sample_correct(static_cast<size_t>(batch_size));
  std::vector<float> grad_cur;
  std::vector<float> grad_prev;
  std::vector<float> deriv_all;
  std::vector<int32_t> resolved_first_mapping(static_cast<size_t>(lm.num_slots));

  for (int64_t epoch = 0; epoch < epochs; ++epoch) {
    torch::Tensor permutation = torch::randperm(
        n_samples, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    const int64_t* perm = permutation.data_ptr<int64_t>();
    const int64_t lr_drops = epoch / lr_decay_steps;
    const float lr = learning_rate * std::pow(gamma, static_cast<float>(lr_drops));
    double epoch_loss = 0.0;
    int64_t epoch_correct = 0;

    for (int64_t batch_start = 0; batch_start < n_samples; batch_start += batch_size) {
      const int64_t bs = std::min<int64_t>(batch_size, n_samples - batch_start);

      for (auto& layer : layers) clamp_layer(layer);
      if (first_learnable) {
        resolve_first_mapping(lm, resolved_first_mapping);
        layers[0].mapping = resolved_first_mapping;
      }

      ones_indices.clear();
      ones_offsets[0] = 0;
      for (int64_t b = 0; b < bs; ++b) {
        const int64_t src_idx = perm[batch_start + b];
        const uint8_t* src = x_ptr + src_idx * input_size0;
        uint8_t* dst = x_batch.data() + b * input_size0;
        std::memcpy(dst, src, static_cast<size_t>(input_size0));
        for (int64_t f = 0; f < input_size0; ++f) {
          if (src[f] != 0) ones_indices.push_back(static_cast<int32_t>(f));
        }
        ones_offsets[static_cast<size_t>(b + 1)] = static_cast<int64_t>(ones_indices.size());
        batch_y[static_cast<size_t>(b)] = y_ptr[src_idx];
      }

      for (int l = 0; l < L; ++l) {
        const LayerState& layer = layers[l];
        const uint8_t* input = l == 0 ? x_batch.data() : activations[static_cast<size_t>(l - 1)].data();
        const int64_t input_width = layer.input_size;
        uint8_t* out = activations[static_cast<size_t>(l)].data();
        uint32_t* addr_out = addresses[static_cast<size_t>(l)].data();
        at::parallel_for(0, bs, 16, [&](int64_t begin, int64_t end) {
          for (int64_t b = begin; b < end; ++b) {
            const uint8_t* row = input + b * input_width;
            for (int j = 0; j < layer.num_luts; ++j) {
              const int32_t* m = layer.mapping.data() + static_cast<int64_t>(j) * fan_in;
              uint32_t addr = 0;
              for (int bit = 0; bit < fan_in; ++bit)
                addr |= static_cast<uint32_t>(row[m[bit]] != 0) << bit;
              addr_out[static_cast<size_t>(b * layer.num_luts + j)] = addr;
              out[static_cast<size_t>(b * layer.num_luts + j)] =
                  static_cast<uint8_t>(layer.luts[static_cast<size_t>(static_cast<int64_t>(j) * lut_size + addr)] > 0.0f);
            }
          }
        });
      }

      at::parallel_for(0, bs, 32, [&](int64_t begin, int64_t end) {
        std::vector<int> counts(static_cast<size_t>(num_classes));
        const uint8_t* last = activations.back().data();
        for (int64_t b = begin; b < end; ++b) {
          std::fill(counts.begin(), counts.end(), 0);
          for (int j = 0; j < last_n; ++j)
            counts[static_cast<size_t>(j / luts_per_class)] += last[b * last_n + j];
          int max_count = counts[0];
          int pred = 0;
          for (int c = 1; c < num_classes; ++c) {
            if (counts[static_cast<size_t>(c)] > max_count) {
              max_count = counts[static_cast<size_t>(c)];
              pred = c;
            }
          }
          float denom = 0.0f;
          for (int c = 0; c < num_classes; ++c)
            denom += exp_neg_diff[static_cast<size_t>(max_count - counts[static_cast<size_t>(c)])];
          const int64_t label = batch_y[static_cast<size_t>(b)];
          float* cg = class_grad.data() + b * num_classes;
          for (int c = 0; c < num_classes; ++c) {
            const float p = exp_neg_diff[static_cast<size_t>(max_count - counts[static_cast<size_t>(c)])] / denom;
            cg[c] = (p - (label == c ? 1.0f : 0.0f)) /
                    (static_cast<float>(bs) * tau);
            if (label == c) sample_loss[static_cast<size_t>(b)] = -std::log(std::max(p, 1e-30f));
          }
          sample_correct[static_cast<size_t>(b)] = static_cast<uint8_t>(pred == label);
        }
      });

      for (int64_t b = 0; b < bs; ++b) {
        epoch_loss += sample_loss[static_cast<size_t>(b)];
        epoch_correct += sample_correct[static_cast<size_t>(b)];
      }

      grad_cur.resize(static_cast<size_t>(bs * last_n));
      at::parallel_for(0, bs, 32, [&](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; ++b) {
          for (int j = 0; j < last_n; ++j)
            grad_cur[static_cast<size_t>(b * last_n + j)] =
                class_grad[static_cast<size_t>(b * num_classes + j / luts_per_class)];
        }
      });

      for (int l = L - 1; l >= 0; --l) {
        LayerState& layer = layers[l];
        make_layer_derivatives<FixedN>(layer, deriv_all, alpha, beta);
        const int64_t per_lut_deriv = static_cast<int64_t>(fan_in) * half;
        const bool need_prev = l > 0;

        if (need_prev) {
          grad_prev.assign(static_cast<size_t>(bs * layer.input_size), 0.0f);
          at::parallel_for(0, bs, 8, [&](int64_t begin, int64_t end) {
            for (int64_t b = begin; b < end; ++b) {
              float* gp = grad_prev.data() + b * layer.input_size;
              for (int j = 0; j < layer.num_luts; ++j) {
                const float g = grad_cur[static_cast<size_t>(b * layer.num_luts + j)];
                const uint32_t addr = addresses[static_cast<size_t>(l)][static_cast<size_t>(b * layer.num_luts + j)];
                const float* d = deriv_all.data() + static_cast<int64_t>(j) * per_lut_deriv;
                const int32_t* m = layer.mapping.data() + static_cast<int64_t>(j) * fan_in;
                for (int bit = 0; bit < fan_in; ++bit) {
                  const int q = compact_addr_without_bit(addr, bit);
                  gp[m[bit]] += d[static_cast<int64_t>(bit) * half + q] * g;
                }
              }
            }
          });
        }

        if (l == 0 && first_learnable) {
          at::parallel_for(0, layer.num_luts, 1, [&](int64_t begin, int64_t end) {
            std::vector<float> lut_grad(static_cast<size_t>(lut_size));
            std::vector<float> map_grad(static_cast<size_t>(static_cast<int64_t>(fan_in) * input_size0));
            std::vector<float> sum_mg(static_cast<size_t>(fan_in));
            std::vector<float> mg(static_cast<size_t>(fan_in));
            for (int64_t j64 = begin; j64 < end; ++j64) {
              const int j = static_cast<int>(j64);
              std::fill(lut_grad.begin(), lut_grad.end(), 0.0f);
              std::fill(sum_mg.begin(), sum_mg.end(), 0.0f);
              const float* d = deriv_all.data() + static_cast<int64_t>(j) * per_lut_deriv;
              for (int64_t b = 0; b < bs; ++b) {
                const float g = grad_cur[static_cast<size_t>(b * layer.num_luts + j)];
                const uint32_t addr = addresses[0][static_cast<size_t>(b * layer.num_luts + j)];
                lut_grad[addr] += g;
                for (int bit = 0; bit < fan_in; ++bit) {
                  const int q = compact_addr_without_bit(addr, bit);
                  sum_mg[static_cast<size_t>(bit)] += d[static_cast<int64_t>(bit) * half + q] * g;
                }
              }
              for (int bit = 0; bit < fan_in; ++bit) {
                float* gr = map_grad.data() + static_cast<int64_t>(bit) * input_size0;
                std::fill(gr, gr + input_size0, -sum_mg[static_cast<size_t>(bit)]);
              }
              for (int64_t b = 0; b < bs; ++b) {
                const float g = grad_cur[static_cast<size_t>(b * layer.num_luts + j)];
                const uint32_t addr = addresses[0][static_cast<size_t>(b * layer.num_luts + j)];
                for (int bit = 0; bit < fan_in; ++bit) {
                  const int q = compact_addr_without_bit(addr, bit);
                  mg[static_cast<size_t>(bit)] = d[static_cast<int64_t>(bit) * half + q] * g;
                }
                const int64_t ob = ones_offsets[static_cast<size_t>(b)];
                const int64_t oe = ones_offsets[static_cast<size_t>(b + 1)];
                for (int64_t p = ob; p < oe; ++p) {
                  const int32_t f = ones_indices[static_cast<size_t>(p)];
                  for (int bit = 0; bit < fan_in; ++bit)
                    map_grad[static_cast<int64_t>(bit) * input_size0 + f] += 2.0f * mg[static_cast<size_t>(bit)];
                }
              }
              float* lut = layer.luts.data() + static_cast<int64_t>(j) * lut_size;
              for (int64_t a = 0; a < lut_size; ++a) lut[a] -= lr * lut_grad[static_cast<size_t>(a)];
              for (int bit = 0; bit < fan_in; ++bit) {
                const int64_t slot_idx = static_cast<int64_t>(j) * fan_in + bit;
                float* w = lm.slot(slot_idx);
                const float* gr = map_grad.data() + static_cast<int64_t>(bit) * input_size0;
                for (int64_t f = 0; f < input_size0; ++f) w[f] -= lr * gr[f];
              }
            }
          });
        } else {
          at::parallel_for(0, layer.num_luts, 1, [&](int64_t begin, int64_t end) {
            std::vector<float> lut_grad(static_cast<size_t>(lut_size));
            for (int64_t j64 = begin; j64 < end; ++j64) {
              const int j = static_cast<int>(j64);
              std::fill(lut_grad.begin(), lut_grad.end(), 0.0f);
              for (int64_t b = 0; b < bs; ++b) {
                const float g = grad_cur[static_cast<size_t>(b * layer.num_luts + j)];
                const uint32_t addr = addresses[static_cast<size_t>(l)][static_cast<size_t>(b * layer.num_luts + j)];
                lut_grad[addr] += g;
              }
              float* lut = layer.luts.data() + static_cast<int64_t>(j) * lut_size;
              for (int64_t a = 0; a < lut_size; ++a) lut[a] -= lr * lut_grad[static_cast<size_t>(a)];
            }
          });
        }

        if (need_prev) grad_cur.swap(grad_prev);
      }
    }

    if (verbose) {
      std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                << ", Train Loss: " << (epoch_loss / static_cast<double>(n_samples))
                << ", Train Accuracy: "
                << (static_cast<double>(epoch_correct) / static_cast<double>(n_samples))
                << ", LR: " << lr << std::endl;
    }

    if (PyErr_CheckSignals() != 0) {
      throw py::error_already_set();
    }
  }

  for (auto& layer : layers) clamp_layer(layer);
  if (first_learnable) {
    resolve_first_mapping(lm, resolved_first_mapping);
    layers[0].mapping = resolved_first_mapping;
  }

  std::vector<torch::Tensor> out;
  out.reserve(static_cast<size_t>(2 * L));
  for (int l = 0; l < L; ++l) {
    LayerState& layer = layers[l];
    auto bl = torch::empty({layer.num_luts, lut_size},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    auto mp = torch::empty({layer.num_luts, fan_in},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    uint8_t* bp = bl.data_ptr<uint8_t>();
    int32_t* mpp = mp.data_ptr<int32_t>();
    for (int64_t i = 0; i < static_cast<int64_t>(layer.num_luts) * lut_size; ++i)
      bp[i] = static_cast<uint8_t>(layer.luts[static_cast<size_t>(i)] > 0.0f);
    std::memcpy(mpp, layer.mapping.data(), layer.mapping.size() * sizeof(int32_t));
    out.push_back(bl);
    out.push_back(mp);
  }
  return out;
}

template <int FixedN>
torch::Tensor predict_multi_impl(
    torch::Tensor x,
    std::vector<torch::Tensor> binary_luts,
    std::vector<torch::Tensor> mappings,
    int runtime_n,
    int64_t num_classes64) {
  const int fan_in = FixedN > 0 ? FixedN : runtime_n;
  TORCH_CHECK(x.device().is_cpu() && x.is_contiguous() && x.scalar_type() == torch::kUInt8 && x.dim() == 2,
              "x must be contiguous CPU uint8 [num_samples, input_size]");
  TORCH_CHECK(binary_luts.size() >= 2 && binary_luts.size() == mappings.size(),
              "binary_luts and mappings must contain the same number of >=2 layers");
  TORCH_CHECK(num_classes64 >= 2, "num_classes must be >= 2");
  const int L = static_cast<int>(binary_luts.size());
  const int64_t lut_size = lut_size_for_n(fan_in);
  int64_t prev = x.size(1);
  std::vector<int> nl(static_cast<size_t>(L));
  for (int l = 0; l < L; ++l) {
    const auto& bl = binary_luts[static_cast<size_t>(l)];
    const auto& mp = mappings[static_cast<size_t>(l)];
    TORCH_CHECK(bl.device().is_cpu() && bl.is_contiguous() && bl.scalar_type() == torch::kUInt8 &&
                    bl.dim() == 2 && bl.size(1) == lut_size,
                "binary LUT tensor has wrong shape/type");
    nl[static_cast<size_t>(l)] = static_cast<int>(bl.size(0));
    check_mapping_tensor(mp, bl.size(0), fan_in, prev);
    prev = bl.size(0);
  }
  TORCH_CHECK(nl.back() % num_classes64 == 0,
              "last layer num_luts must be divisible by num_classes");

  const int64_t ns = x.size(0);
  std::vector<uint8_t> cur;
  std::vector<uint8_t> next;
  const uint8_t* input_ptr = x.data_ptr<uint8_t>();
  int64_t input_width = x.size(1);

  for (int l = 0; l < L; ++l) {
    const int nout = nl[static_cast<size_t>(l)];
    next.resize(static_cast<size_t>(ns * nout));
    const uint8_t* lut = binary_luts[static_cast<size_t>(l)].data_ptr<uint8_t>();
    const int32_t* map = mappings[static_cast<size_t>(l)].data_ptr<int32_t>();
    const uint8_t* in = l == 0 ? input_ptr : cur.data();
    at::parallel_for(0, ns, 32, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        const uint8_t* row = in + b * input_width;
        for (int j = 0; j < nout; ++j) {
          uint32_t addr = 0;
          const int32_t* m = map + static_cast<int64_t>(j) * fan_in;
          for (int bit = 0; bit < fan_in; ++bit)
            addr |= static_cast<uint32_t>(row[m[bit]] != 0) << bit;
          next[static_cast<size_t>(b * nout + j)] =
              lut[static_cast<size_t>(static_cast<int64_t>(j) * lut_size + addr)];
        }
      }
    });
    cur.swap(next);
    input_width = nout;
  }

  const int num_classes = static_cast<int>(num_classes64);
  const int last_n = nl.back();
  const int per_class = last_n / num_classes;
  auto pred = torch::empty({ns}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* pp = pred.data_ptr<int64_t>();
  at::parallel_for(0, ns, 64, [&](int64_t begin, int64_t end) {
    std::vector<int> counts(static_cast<size_t>(num_classes));
    for (int64_t b = begin; b < end; ++b) {
      std::fill(counts.begin(), counts.end(), 0);
      for (int j = 0; j < last_n; ++j)
        counts[static_cast<size_t>(j / per_class)] += cur[static_cast<size_t>(b * last_n + j)];
      int best = 0;
      for (int c = 1; c < num_classes; ++c)
        if (counts[static_cast<size_t>(c)] > counts[static_cast<size_t>(best)]) best = c;
      pp[b] = best;
    }
  });
  return pred;
}

#define DEFINE_MULTI(N) \
std::vector<torch::Tensor> train_multi_lut##N( \
    torch::Tensor x, torch::Tensor y, std::vector<torch::Tensor> initial_luts, \
    torch::Tensor first_mapping_weights, std::vector<torch::Tensor> fixed_mappings, \
    float tau, float learning_rate, int64_t epochs, int64_t batch_size, \
    int64_t lr_decay_steps, float gamma, int64_t num_classes, \
    bool first_learnable, bool verbose) { \
  return train_multi_impl<N>(x, y, initial_luts, first_mapping_weights, fixed_mappings, N, \
      tau, learning_rate, epochs, batch_size, lr_decay_steps, gamma, num_classes, \
      first_learnable, verbose); \
} \
torch::Tensor predict_multi_lut##N( \
    torch::Tensor x, std::vector<torch::Tensor> binary_luts, \
    std::vector<torch::Tensor> mappings, int64_t num_classes) { \
  return predict_multi_impl<N>(x, binary_luts, mappings, N, num_classes); \
}

DEFINE_MULTI(2)
DEFINE_MULTI(3)
DEFINE_MULTI(4)
DEFINE_MULTI(5)
DEFINE_MULTI(6)
DEFINE_MULTI(8)

std::vector<torch::Tensor> train_multi_general(
    torch::Tensor x, torch::Tensor y, std::vector<torch::Tensor> initial_luts,
    torch::Tensor first_mapping_weights, std::vector<torch::Tensor> fixed_mappings,
    int64_t n, float tau, float learning_rate, int64_t epochs, int64_t batch_size,
    int64_t lr_decay_steps, float gamma, int64_t num_classes,
    bool first_learnable, bool verbose) {
  TORCH_CHECK(n >= 1 && n < 31, "n must be in [1, 30]");
  return train_multi_impl<0>(x, y, initial_luts, first_mapping_weights, fixed_mappings,
      static_cast<int>(n), tau, learning_rate, epochs, batch_size, lr_decay_steps,
      gamma, num_classes, first_learnable, verbose);
}

torch::Tensor predict_multi_general(
    torch::Tensor x, std::vector<torch::Tensor> binary_luts,
    std::vector<torch::Tensor> mappings, int64_t n, int64_t num_classes) {
  TORCH_CHECK(n >= 1 && n < 31, "n must be in [1, 30]");
  return predict_multi_impl<0>(x, binary_luts, mappings, static_cast<int>(n), num_classes);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define BIND_MULTI_TRAIN(N) \
  m.def("train_multi_lut" #N, &train_multi_lut##N, \
      py::arg("x"), py::arg("y"), py::arg("initial_luts"), \
      py::arg("first_mapping_weights"), py::arg("fixed_mappings"), \
      py::arg("tau"), py::arg("learning_rate"), py::arg("epochs"), \
      py::arg("batch_size") = 512, py::arg("lr_decay_steps") = 8, \
      py::arg("gamma") = 0.1f, py::arg("num_classes") = 2, \
      py::arg("first_learnable") = true, py::arg("verbose") = true);

  BIND_MULTI_TRAIN(2)
  BIND_MULTI_TRAIN(3)
  BIND_MULTI_TRAIN(4)
  BIND_MULTI_TRAIN(5)
  BIND_MULTI_TRAIN(6)
  BIND_MULTI_TRAIN(8)
#undef BIND_MULTI_TRAIN

  m.def("predict_multi_lut2", &predict_multi_lut2,
        py::arg("x"), py::arg("binary_luts"), py::arg("mappings"), py::arg("num_classes") = 2);
  m.def("predict_multi_lut3", &predict_multi_lut3,
        py::arg("x"), py::arg("binary_luts"), py::arg("mappings"), py::arg("num_classes") = 2);
  m.def("predict_multi_lut4", &predict_multi_lut4,
        py::arg("x"), py::arg("binary_luts"), py::arg("mappings"), py::arg("num_classes") = 2);
  m.def("predict_multi_lut5", &predict_multi_lut5,
        py::arg("x"), py::arg("binary_luts"), py::arg("mappings"), py::arg("num_classes") = 2);
  m.def("predict_multi_lut6", &predict_multi_lut6,
        py::arg("x"), py::arg("binary_luts"), py::arg("mappings"), py::arg("num_classes") = 2);
  m.def("predict_multi_lut8", &predict_multi_lut8,
        py::arg("x"), py::arg("binary_luts"), py::arg("mappings"), py::arg("num_classes") = 2);

  m.def("train_multi_general", &train_multi_general,
      py::arg("x"), py::arg("y"), py::arg("initial_luts"),
      py::arg("first_mapping_weights"), py::arg("fixed_mappings"), py::arg("n"),
      py::arg("tau"), py::arg("learning_rate"), py::arg("epochs"),
      py::arg("batch_size") = 512, py::arg("lr_decay_steps") = 8,
      py::arg("gamma") = 0.1f, py::arg("num_classes") = 2,
      py::arg("first_learnable") = true, py::arg("verbose") = true);

  m.def("predict_multi_general", &predict_multi_general,
      py::arg("x"), py::arg("binary_luts"), py::arg("mappings"), py::arg("n"),
      py::arg("num_classes") = 2);
}
