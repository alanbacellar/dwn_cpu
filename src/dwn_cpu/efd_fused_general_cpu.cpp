#include <torch/extension.h>
#include <ATen/Parallel.h>

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

inline bool packed_feature(const uint64_t* row, int32_t feature) {
  const uint32_t f = static_cast<uint32_t>(feature);
  return ((row[f >> 6] >> (f & 63u)) & 1ull) != 0ull;
}

struct ModelStorage {
  int64_t input_size;
  int num_luts;
  int fan_in;
  int64_t lut_size;
  int64_t block_stride;
  std::vector<float> data;

  ModelStorage(int64_t input_size_, int num_luts_, int fan_in_)
      : input_size(input_size_),
        num_luts(num_luts_),
        fan_in(fan_in_),
        lut_size(lut_size_for_n(fan_in_)),
        block_stride(lut_size + static_cast<int64_t>(fan_in) * input_size),
        data(static_cast<size_t>(static_cast<int64_t>(num_luts) * block_stride), 0.0f) {}

  inline float* block(int lut) {
    return data.data() + static_cast<int64_t>(lut) * block_stride;
  }
  inline const float* block(int lut) const {
    return data.data() + static_cast<int64_t>(lut) * block_stride;
  }
  inline float* lut(int lut) { return block(lut); }
  inline const float* lut(int lut) const { return block(lut); }
  inline float* mapping_weights(int lut, int bit) {
    return block(lut) + lut_size + static_cast<int64_t>(bit) * input_size;
  }
  inline const float* mapping_weights(int lut, int bit) const {
    return block(lut) + lut_size + static_cast<int64_t>(bit) * input_size;
  }
};

inline void make_efd_derivatives_runtime(
    const float* lut,
    float* deriv,
    int fan_in,
    float alpha,
    float beta) {
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

    for (int q = 0; q < half; ++q) {
      v[q] *= alpha;
    }
  }
}

template <int N>
inline void make_efd_derivatives_fixed(
    const float* lut,
    float* deriv,
    float alpha,
    float beta) {
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

    for (int q = 0; q < half; ++q) {
      v[q] *= alpha;
    }
  }
}

void check_train_inputs(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& initial_luts,
    const torch::Tensor& initial_mapping_weights,
    int fan_in,
    float tau,
    float learning_rate,
    int64_t epochs,
    int64_t batch_size,
    int64_t lr_decay_steps,
    float gamma,
    int64_t num_classes,
    int64_t num_luts) {
  TORCH_CHECK(x.device().is_cpu() && y.device().is_cpu() &&
                  initial_luts.device().is_cpu() &&
                  initial_mapping_weights.device().is_cpu(),
              "all tensors must be on CPU");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous() &&
                  initial_luts.is_contiguous() &&
                  initial_mapping_weights.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kUInt8, "x must be torch.uint8");
  TORCH_CHECK(y.scalar_type() == torch::kInt64, "y must be torch.int64");
  TORCH_CHECK(initial_luts.scalar_type() == torch::kFloat32,
              "initial_luts must be torch.float32");
  TORCH_CHECK(initial_mapping_weights.scalar_type() == torch::kFloat32,
              "initial_mapping_weights must be torch.float32");
  TORCH_CHECK(x.dim() == 2, "x must have shape [num_samples, input_size]");
  TORCH_CHECK(y.dim() == 1 && y.size(0) == x.size(0),
              "y must have shape [num_samples]");
  TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0, "x must be non-empty");
  TORCH_CHECK(x.size(1) <= std::numeric_limits<int32_t>::max(),
              "input_size is too large");
  TORCH_CHECK(num_classes >= 2, "num_classes must be >= 2");
  TORCH_CHECK(num_luts > 0, "num_luts must be > 0");
  TORCH_CHECK(num_luts % num_classes == 0,
              "num_luts must be divisible by num_classes");
  TORCH_CHECK(num_luts <= std::numeric_limits<int>::max(), "num_luts is too large");
  TORCH_CHECK(num_classes <= std::numeric_limits<int>::max(), "num_classes is too large");
  TORCH_CHECK(initial_luts.dim() == 2 && initial_luts.size(0) == num_luts &&
                  initial_luts.size(1) == lut_size_for_n(fan_in),
              "initial_luts has wrong shape for num_luts and n");
  TORCH_CHECK(initial_mapping_weights.dim() == 2 &&
                  initial_mapping_weights.size(0) == x.size(1) &&
                  initial_mapping_weights.size(1) == num_luts * fan_in,
              "initial_mapping_weights must have shape [input_size, num_luts*n]");
  TORCH_CHECK(tau > 0.0f && std::isfinite(tau), "tau must be finite and > 0");
  TORCH_CHECK(learning_rate > 0.0f && std::isfinite(learning_rate),
              "learning_rate must be finite and > 0");
  TORCH_CHECK(epochs >= 0, "epochs must be >= 0");
  TORCH_CHECK(batch_size > 0, "batch_size must be > 0");
  TORCH_CHECK(lr_decay_steps > 0, "lr_decay_steps must be > 0");
  TORCH_CHECK(gamma > 0.0f && std::isfinite(gamma), "gamma must be finite and > 0");

  const int64_t* yp = y.data_ptr<int64_t>();
  for (int64_t i = 0; i < y.size(0); ++i) {
    TORCH_CHECK(yp[i] >= 0 && yp[i] < num_classes,
                "y contains a class outside [0, num_classes)");
  }
}

void prepare_batch(
    ModelStorage& model,
    std::vector<int32_t>& mapping) {
  const int64_t input_size = model.input_size;
  const int fan_in = model.fan_in;
  const int64_t lut_size = model.lut_size;
  const int num_luts = model.num_luts;

  at::parallel_for(0, num_luts, 1, [&](int64_t begin, int64_t end) {
    for (int64_t j64 = begin; j64 < end; ++j64) {
      const int j = static_cast<int>(j64);
      float* lut = model.lut(j);
      for (int64_t a = 0; a < lut_size; ++a) {
        lut[a] = std::max(-1.0f, std::min(1.0f, lut[a]));
      }

      for (int bit = 0; bit < fan_in; ++bit) {
        const float* w = model.mapping_weights(j, bit);
        int32_t best = 0;
        float best_value = w[0];
        for (int64_t f = 1; f < input_size; ++f) {
          if (w[f] > best_value) {
            best_value = w[f];
            best = static_cast<int32_t>(f);
          }
        }
        mapping[static_cast<int64_t>(j) * fan_in + bit] = best;
      }
    }
  });
}

template <int FixedN>
std::vector<torch::Tensor> train_impl(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor initial_luts,
    torch::Tensor initial_mapping_weights,
    int runtime_n,
    float tau,
    float learning_rate,
    int64_t epochs,
    int64_t batch_size,
    int64_t lr_decay_steps,
    float gamma,
    int64_t num_classes64,
    int64_t num_luts64,
    int64_t start_epoch,
    bool verbose) {
  const int fan_in = FixedN > 0 ? FixedN : runtime_n;
  check_train_inputs(x, y, initial_luts, initial_mapping_weights, fan_in,
                     tau, learning_rate, epochs, batch_size, lr_decay_steps,
                     gamma, num_classes64, num_luts64);
  TORCH_CHECK(start_epoch >= 0, "start_epoch must be >= 0");

  const int num_classes = static_cast<int>(num_classes64);
  const int num_luts = static_cast<int>(num_luts64);
  const int luts_per_class = num_luts / num_classes;
  const int64_t n_samples = x.size(0);
  const int64_t input_size = x.size(1);
  const int64_t lut_size = lut_size_for_n(fan_in);
  const int half = 1 << (fan_in - 1);
  const int64_t num_slots = static_cast<int64_t>(num_luts) * fan_in;
  const int64_t packed_words = (input_size + 63) / 64;
  const float alpha = 0.5f * std::pow(0.75f, static_cast<float>(fan_in - 1));
  constexpr float beta = 1.0f / 3.0f;

  const uint8_t* x_ptr = x.data_ptr<uint8_t>();
  const int64_t* y_ptr = y.data_ptr<int64_t>();
  const float* init_lut_ptr = initial_luts.data_ptr<float>();
  const float* init_map_ptr = initial_mapping_weights.data_ptr<float>();

  ModelStorage model(input_size, num_luts, fan_in);
  for (int j = 0; j < num_luts; ++j) {
    std::memcpy(model.lut(j),
                init_lut_ptr + static_cast<int64_t>(j) * lut_size,
                static_cast<size_t>(lut_size) * sizeof(float));
    for (int bit = 0; bit < fan_in; ++bit) {
      const int64_t slot = static_cast<int64_t>(j) * fan_in + bit;
      float* dst = model.mapping_weights(j, bit);
      for (int64_t f = 0; f < input_size; ++f) {
        dst[f] = init_map_ptr[f * num_slots + slot];
      }
    }
  }

  std::vector<float> exp_neg_diff(static_cast<size_t>(luts_per_class + 1));
  for (int d = 0; d <= luts_per_class; ++d) {
    exp_neg_diff[static_cast<size_t>(d)] =
        std::exp(-static_cast<float>(d) / tau);
  }

  std::vector<int32_t> mapping(static_cast<size_t>(num_slots));
  std::vector<uint64_t> packed_x(static_cast<size_t>(batch_size * packed_words));
  std::vector<int64_t> ones_offsets(static_cast<size_t>(batch_size + 1));
  std::vector<int32_t> ones_indices;
  const int64_t reserve_guess = std::min<int64_t>(
      batch_size * std::min<int64_t>(input_size, 4096) / 2,
      int64_t{1} << 26);
  ones_indices.reserve(static_cast<size_t>(std::max<int64_t>(reserve_guess, 1)));
  std::vector<int64_t> batch_y(static_cast<size_t>(batch_size));
  std::vector<uint32_t> addresses(static_cast<size_t>(batch_size * num_luts));
  std::vector<float> sample_class_grad(static_cast<size_t>(batch_size * num_classes));
  std::vector<float> sample_loss(static_cast<size_t>(batch_size));
  std::vector<uint8_t> sample_correct(static_cast<size_t>(batch_size));

  for (int64_t epoch = 0; epoch < epochs; ++epoch) {
    torch::Tensor permutation = torch::randperm(
        n_samples,
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    const int64_t* perm_ptr = permutation.data_ptr<int64_t>();
    const int64_t global_epoch = start_epoch + epoch;
    const int64_t lr_drops = global_epoch / lr_decay_steps;
    const float lr = learning_rate *
        std::pow(gamma, static_cast<float>(lr_drops));

    double epoch_loss_sum = 0.0;
    int64_t epoch_correct = 0;

    for (int64_t batch_start = 0; batch_start < n_samples; batch_start += batch_size) {
      const int64_t bs = std::min<int64_t>(batch_size, n_samples - batch_start);
      prepare_batch(model, mapping);

      std::fill(packed_x.begin(),
                packed_x.begin() + static_cast<size_t>(bs * packed_words),
                uint64_t{0});
      ones_indices.clear();
      ones_offsets[0] = 0;

      for (int64_t b = 0; b < bs; ++b) {
        const int64_t src_idx = perm_ptr[batch_start + b];
        const uint8_t* src = x_ptr + src_idx * input_size;
        uint64_t* packed_row = packed_x.data() + b * packed_words;
        for (int64_t f = 0; f < input_size; ++f) {
          if (src[f] != 0) {
            packed_row[static_cast<uint64_t>(f) >> 6] |=
                uint64_t{1} << (static_cast<uint64_t>(f) & 63u);
            ones_indices.push_back(static_cast<int32_t>(f));
          }
        }
        ones_offsets[static_cast<size_t>(b + 1)] =
            static_cast<int64_t>(ones_indices.size());
        batch_y[static_cast<size_t>(b)] = y_ptr[src_idx];
      }

      at::parallel_for(0, bs, 32, [&](int64_t begin, int64_t end) {
        std::vector<int> counts(static_cast<size_t>(num_classes));
        for (int64_t b = begin; b < end; ++b) {
          std::fill(counts.begin(), counts.end(), 0);
          const uint64_t* packed_row = packed_x.data() + b * packed_words;

          for (int j = 0; j < num_luts; ++j) {
            uint32_t addr = 0;
            const int32_t* map_row = mapping.data() + static_cast<int64_t>(j) * fan_in;
            for (int bit = 0; bit < fan_in; ++bit) {
              addr |= static_cast<uint32_t>(packed_feature(packed_row, map_row[bit])) << bit;
            }
            addresses[static_cast<size_t>(b * num_luts + j)] = addr;
            counts[static_cast<size_t>(j / luts_per_class)] +=
                model.lut(j)[addr] > 0.0f ? 1 : 0;
          }

          int max_count = counts[0];
          int pred = 0;
          for (int c = 1; c < num_classes; ++c) {
            if (counts[static_cast<size_t>(c)] > max_count) {
              max_count = counts[static_cast<size_t>(c)];
              pred = c;
            }
          }

          float denom = 0.0f;
          for (int c = 0; c < num_classes; ++c) {
            denom += exp_neg_diff[static_cast<size_t>(max_count - counts[static_cast<size_t>(c)])];
          }

          const int64_t label = batch_y[static_cast<size_t>(b)];
          float* grad_row = sample_class_grad.data() + b * num_classes;
          for (int c = 0; c < num_classes; ++c) {
            const float p = exp_neg_diff[
                static_cast<size_t>(max_count - counts[static_cast<size_t>(c)])] / denom;
            grad_row[c] =
                (p - (label == c ? 1.0f : 0.0f)) /
                (static_cast<float>(bs) * tau);
            if (label == c) {
              sample_loss[static_cast<size_t>(b)] =
                  -std::log(std::max(p, 1e-30f));
            }
          }
          sample_correct[static_cast<size_t>(b)] =
              static_cast<uint8_t>(pred == label);
        }
      });

      for (int64_t b = 0; b < bs; ++b) {
        epoch_loss_sum += sample_loss[static_cast<size_t>(b)];
        epoch_correct += sample_correct[static_cast<size_t>(b)];
      }

      at::parallel_for(0, num_luts, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> mapping_grad(
            static_cast<size_t>(static_cast<int64_t>(fan_in) * input_size), 0.0f);
        std::vector<float> deriv(static_cast<size_t>(fan_in * half));
        std::vector<float> lut_grad(static_cast<size_t>(lut_size));
        std::vector<float> sum_mapped_grad(static_cast<size_t>(fan_in));
        std::vector<float> mg(static_cast<size_t>(fan_in));

        for (int64_t j64 = begin; j64 < end; ++j64) {
          const int j = static_cast<int>(j64);
          const int class_idx = j / luts_per_class;
          float* lut = model.lut(j);

          if constexpr (FixedN > 0) {
            make_efd_derivatives_fixed<FixedN>(lut, deriv.data(), alpha, beta);
          } else {
            make_efd_derivatives_runtime(lut, deriv.data(), fan_in, alpha, beta);
          }

          std::fill(lut_grad.begin(), lut_grad.end(), 0.0f);
          std::fill(sum_mapped_grad.begin(), sum_mapped_grad.end(), 0.0f);

          for (int64_t b = 0; b < bs; ++b) {
            const float g = sample_class_grad[
                static_cast<size_t>(b * num_classes + class_idx)];
            const uint32_t addr = addresses[static_cast<size_t>(b * num_luts + j)];
            lut_grad[addr] += g;
            for (int bit = 0; bit < fan_in; ++bit) {
              const int q = compact_addr_without_bit(addr, bit);
              sum_mapped_grad[static_cast<size_t>(bit)] +=
                  deriv[static_cast<size_t>(bit * half + q)] * g;
            }
          }

          for (int bit = 0; bit < fan_in; ++bit) {
            float* grad_row = mapping_grad.data() + static_cast<int64_t>(bit) * input_size;
            std::fill(grad_row, grad_row + input_size,
                      -sum_mapped_grad[static_cast<size_t>(bit)]);
          }

          for (int64_t b = 0; b < bs; ++b) {
            const float g = sample_class_grad[
                static_cast<size_t>(b * num_classes + class_idx)];
            const uint32_t addr = addresses[static_cast<size_t>(b * num_luts + j)];
            for (int bit = 0; bit < fan_in; ++bit) {
              const int q = compact_addr_without_bit(addr, bit);
              mg[static_cast<size_t>(bit)] =
                  deriv[static_cast<size_t>(bit * half + q)] * g;
            }

            const int64_t ob = ones_offsets[static_cast<size_t>(b)];
            const int64_t oe = ones_offsets[static_cast<size_t>(b + 1)];
            for (int64_t pos = ob; pos < oe; ++pos) {
              const int32_t f = ones_indices[static_cast<size_t>(pos)];
              for (int bit = 0; bit < fan_in; ++bit) {
                mapping_grad[static_cast<int64_t>(bit) * input_size + f] +=
                    2.0f * mg[static_cast<size_t>(bit)];
              }
            }
          }

          for (int64_t a = 0; a < lut_size; ++a) {
            lut[a] -= lr * lut_grad[static_cast<size_t>(a)];
          }
          for (int bit = 0; bit < fan_in; ++bit) {
            float* w = model.mapping_weights(j, bit);
            const float* grad_row = mapping_grad.data() + static_cast<int64_t>(bit) * input_size;
            for (int64_t f = 0; f < input_size; ++f) {
              w[f] -= lr * grad_row[f];
            }
          }
        }
      });
    }

    if (verbose) {
      std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                << ", Train Loss: " << (epoch_loss_sum / static_cast<double>(n_samples))
                << ", Train Accuracy: "
                << (static_cast<double>(epoch_correct) / static_cast<double>(n_samples))
                << ", LR: " << lr << std::endl;
    }

    if (PyErr_CheckSignals() != 0) {
      throw py::error_already_set();
    }
  }

  prepare_batch(model, mapping);

  auto final_luts = torch::empty(
      {num_luts, lut_size},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto final_mapping_weights = torch::empty(
      {input_size, num_slots},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto binary_luts = torch::empty(
      {num_luts, lut_size},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  auto final_mapping = torch::empty(
      {num_luts, fan_in},
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

  float* out_float_luts = final_luts.data_ptr<float>();
  float* out_mapping_weights = final_mapping_weights.data_ptr<float>();
  uint8_t* out_luts = binary_luts.data_ptr<uint8_t>();
  int32_t* out_mapping = final_mapping.data_ptr<int32_t>();
  for (int j = 0; j < num_luts; ++j) {
    const float* lut = model.lut(j);
    std::memcpy(out_float_luts + static_cast<int64_t>(j) * lut_size,
                lut, static_cast<size_t>(lut_size) * sizeof(float));
    for (int64_t a = 0; a < lut_size; ++a) {
      out_luts[static_cast<int64_t>(j) * lut_size + a] =
          static_cast<uint8_t>(lut[a] > 0.0f);
    }
    for (int bit = 0; bit < fan_in; ++bit) {
      const int64_t slot = static_cast<int64_t>(j) * fan_in + bit;
      const float* w = model.mapping_weights(j, bit);
      for (int64_t f = 0; f < input_size; ++f) {
        out_mapping_weights[f * num_slots + slot] = w[f];
      }
      out_mapping[static_cast<int64_t>(j) * fan_in + bit] =
          mapping[static_cast<int64_t>(j) * fan_in + bit];
    }
  }

  return {final_luts, final_mapping_weights, binary_luts, final_mapping};
}


void check_fixed_train_inputs(
    const torch::Tensor& x,
    const torch::Tensor& y,
    const torch::Tensor& initial_luts,
    const torch::Tensor& mapping,
    int fan_in,
    float tau,
    float learning_rate,
    int64_t epochs,
    int64_t batch_size,
    int64_t lr_decay_steps,
    float gamma,
    int64_t num_classes,
    int64_t num_luts) {
  TORCH_CHECK(x.device().is_cpu() && y.device().is_cpu() &&
                  initial_luts.device().is_cpu() && mapping.device().is_cpu(),
              "all tensors must be on CPU");
  TORCH_CHECK(x.is_contiguous() && y.is_contiguous() &&
                  initial_luts.is_contiguous() && mapping.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kUInt8, "x must be torch.uint8");
  TORCH_CHECK(y.scalar_type() == torch::kInt64, "y must be torch.int64");
  TORCH_CHECK(initial_luts.scalar_type() == torch::kFloat32,
              "initial_luts must be torch.float32");
  TORCH_CHECK(mapping.scalar_type() == torch::kInt32,
              "mapping must be torch.int32");
  TORCH_CHECK(x.dim() == 2, "x must have shape [num_samples, input_size]");
  TORCH_CHECK(y.dim() == 1 && y.size(0) == x.size(0),
              "y must have shape [num_samples]");
  TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0, "x must be non-empty");
  TORCH_CHECK(num_classes >= 2, "num_classes must be >= 2");
  TORCH_CHECK(num_luts > 0 && num_luts % num_classes == 0,
              "num_luts must be positive and divisible by num_classes");
  TORCH_CHECK(initial_luts.dim() == 2 && initial_luts.size(0) == num_luts &&
                  initial_luts.size(1) == lut_size_for_n(fan_in),
              "initial_luts has wrong shape for num_luts and n");
  TORCH_CHECK(mapping.dim() == 2 && mapping.size(0) == num_luts &&
                  mapping.size(1) == fan_in,
              "mapping must have shape [num_luts, n]");
  TORCH_CHECK(tau > 0.0f && std::isfinite(tau), "tau must be finite and > 0");
  TORCH_CHECK(learning_rate > 0.0f && std::isfinite(learning_rate),
              "learning_rate must be finite and > 0");
  TORCH_CHECK(epochs >= 0, "epochs must be >= 0");
  TORCH_CHECK(batch_size > 0, "batch_size must be > 0");
  TORCH_CHECK(lr_decay_steps > 0, "lr_decay_steps must be > 0");
  TORCH_CHECK(gamma > 0.0f && std::isfinite(gamma), "gamma must be finite and > 0");

  const int64_t* yp = y.data_ptr<int64_t>();
  for (int64_t i = 0; i < y.size(0); ++i) {
    TORCH_CHECK(yp[i] >= 0 && yp[i] < num_classes,
                "y contains a class outside [0, num_classes)");
  }

  const int32_t* mp = mapping.data_ptr<int32_t>();
  const int64_t slots = num_luts * fan_in;
  for (int64_t i = 0; i < slots; ++i) {
    TORCH_CHECK(mp[i] >= 0 && mp[i] < x.size(1),
                "mapping contains an out-of-range feature index");
  }
}

template <int FixedN>
std::vector<torch::Tensor> train_fixed_impl(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor initial_luts,
    torch::Tensor mapping,
    int runtime_n,
    float tau,
    float learning_rate,
    int64_t epochs,
    int64_t batch_size,
    int64_t lr_decay_steps,
    float gamma,
    int64_t num_classes64,
    int64_t num_luts64,
    bool verbose) {
  const int fan_in = FixedN > 0 ? FixedN : runtime_n;
  check_fixed_train_inputs(x, y, initial_luts, mapping, fan_in, tau,
                           learning_rate, epochs, batch_size, lr_decay_steps,
                           gamma, num_classes64, num_luts64);

  const int num_classes = static_cast<int>(num_classes64);
  const int num_luts = static_cast<int>(num_luts64);
  const int luts_per_class = num_luts / num_classes;
  const int64_t n_samples = x.size(0);
  const int64_t input_size = x.size(1);
  const int64_t lut_size = lut_size_for_n(fan_in);

  const uint8_t* x_ptr = x.data_ptr<uint8_t>();
  const int64_t* y_ptr = y.data_ptr<int64_t>();
  const int32_t* map_ptr = mapping.data_ptr<int32_t>();
  const float* init_lut_ptr = initial_luts.data_ptr<float>();

  std::vector<float> luts(static_cast<size_t>(num_luts * lut_size));
  std::memcpy(luts.data(), init_lut_ptr,
              static_cast<size_t>(num_luts * lut_size) * sizeof(float));

  std::vector<float> exp_neg_diff(static_cast<size_t>(luts_per_class + 1));
  for (int d = 0; d <= luts_per_class; ++d) {
    exp_neg_diff[static_cast<size_t>(d)] =
        std::exp(-static_cast<float>(d) / tau);
  }

  std::vector<uint32_t> addresses(static_cast<size_t>(batch_size * num_luts));
  std::vector<float> sample_class_grad(static_cast<size_t>(batch_size * num_classes));
  std::vector<float> sample_loss(static_cast<size_t>(batch_size));
  std::vector<uint8_t> sample_correct(static_cast<size_t>(batch_size));

  for (int64_t epoch = 0; epoch < epochs; ++epoch) {
    torch::Tensor permutation = torch::randperm(
        n_samples,
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    const int64_t* perm_ptr = permutation.data_ptr<int64_t>();
    const int64_t lr_drops = epoch / lr_decay_steps;
    const float lr = learning_rate *
        std::pow(gamma, static_cast<float>(lr_drops));

    double epoch_loss_sum = 0.0;
    int64_t epoch_correct = 0;

    for (int64_t batch_start = 0; batch_start < n_samples; batch_start += batch_size) {
      const int64_t bs = std::min<int64_t>(batch_size, n_samples - batch_start);

      at::parallel_for(0, num_luts, 1, [&](int64_t begin, int64_t end) {
        for (int64_t j = begin; j < end; ++j) {
          float* lut = luts.data() + j * lut_size;
          for (int64_t a = 0; a < lut_size; ++a) {
            lut[a] = std::max(-1.0f, std::min(1.0f, lut[a]));
          }
        }
      });

      at::parallel_for(0, bs, 32, [&](int64_t begin, int64_t end) {
        std::vector<int> counts(static_cast<size_t>(num_classes));
        for (int64_t b = begin; b < end; ++b) {
          std::fill(counts.begin(), counts.end(), 0);
          const int64_t src_idx = perm_ptr[batch_start + b];
          const uint8_t* row = x_ptr + src_idx * input_size;

          for (int j = 0; j < num_luts; ++j) {
            uint32_t addr = 0;
            const int32_t* m = map_ptr + static_cast<int64_t>(j) * fan_in;
            for (int bit = 0; bit < fan_in; ++bit) {
              addr |= static_cast<uint32_t>(row[m[bit]] != 0) << bit;
            }
            addresses[static_cast<size_t>(b * num_luts + j)] = addr;
            counts[static_cast<size_t>(j / luts_per_class)] +=
                luts[static_cast<size_t>(static_cast<int64_t>(j) * lut_size + addr)] > 0.0f ? 1 : 0;
          }

          int max_count = counts[0];
          int pred = 0;
          for (int c = 1; c < num_classes; ++c) {
            if (counts[static_cast<size_t>(c)] > max_count) {
              max_count = counts[static_cast<size_t>(c)];
              pred = c;
            }
          }

          float denom = 0.0f;
          for (int c = 0; c < num_classes; ++c) {
            denom += exp_neg_diff[static_cast<size_t>(max_count - counts[static_cast<size_t>(c)])];
          }

          const int64_t label = y_ptr[src_idx];
          float* grad_row = sample_class_grad.data() + b * num_classes;
          for (int c = 0; c < num_classes; ++c) {
            const float p = exp_neg_diff[
                static_cast<size_t>(max_count - counts[static_cast<size_t>(c)])] / denom;
            grad_row[c] =
                (p - (label == c ? 1.0f : 0.0f)) /
                (static_cast<float>(bs) * tau);
            if (label == c) {
              sample_loss[static_cast<size_t>(b)] =
                  -std::log(std::max(p, 1e-30f));
            }
          }
          sample_correct[static_cast<size_t>(b)] =
              static_cast<uint8_t>(pred == label);
        }
      });

      for (int64_t b = 0; b < bs; ++b) {
        epoch_loss_sum += sample_loss[static_cast<size_t>(b)];
        epoch_correct += sample_correct[static_cast<size_t>(b)];
      }

      at::parallel_for(0, num_luts, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> lut_grad(static_cast<size_t>(lut_size));
        for (int64_t j64 = begin; j64 < end; ++j64) {
          const int j = static_cast<int>(j64);
          const int class_idx = j / luts_per_class;
          std::fill(lut_grad.begin(), lut_grad.end(), 0.0f);

          for (int64_t b = 0; b < bs; ++b) {
            const float g = sample_class_grad[
                static_cast<size_t>(b * num_classes + class_idx)];
            const uint32_t addr = addresses[static_cast<size_t>(b * num_luts + j)];
            lut_grad[addr] += g;
          }

          float* lut = luts.data() + static_cast<int64_t>(j) * lut_size;
          for (int64_t a = 0; a < lut_size; ++a) {
            lut[a] -= lr * lut_grad[static_cast<size_t>(a)];
          }
        }
      });
    }

    if (verbose) {
      std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                << ", Train Loss: " << (epoch_loss_sum / static_cast<double>(n_samples))
                << ", Train Accuracy: "
                << (static_cast<double>(epoch_correct) / static_cast<double>(n_samples))
                << ", LR: " << lr << std::endl;
    }

    if (PyErr_CheckSignals() != 0) {
      throw py::error_already_set();
    }
  }

  at::parallel_for(0, num_luts, 1, [&](int64_t begin, int64_t end) {
    for (int64_t j = begin; j < end; ++j) {
      float* lut = luts.data() + j * lut_size;
      for (int64_t a = 0; a < lut_size; ++a) {
        lut[a] = std::max(-1.0f, std::min(1.0f, lut[a]));
      }
    }
  });

  auto binary_luts = torch::empty(
      {num_luts, lut_size},
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  uint8_t* out_luts = binary_luts.data_ptr<uint8_t>();
  for (int64_t i = 0; i < static_cast<int64_t>(num_luts) * lut_size; ++i) {
    out_luts[i] = static_cast<uint8_t>(luts[static_cast<size_t>(i)] > 0.0f);
  }

  return {binary_luts, mapping};
}

void check_predict_inputs(
    const torch::Tensor& x,
    const torch::Tensor& binary_luts,
    const torch::Tensor& mapping,
    int fan_in,
    int64_t num_classes,
    int64_t num_luts) {
  TORCH_CHECK(x.device().is_cpu() && binary_luts.device().is_cpu() &&
                  mapping.device().is_cpu(),
              "all tensors must be on CPU");
  TORCH_CHECK(x.is_contiguous() && binary_luts.is_contiguous() &&
                  mapping.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kUInt8, "x must be torch.uint8");
  TORCH_CHECK(binary_luts.scalar_type() == torch::kUInt8,
              "binary_luts must be torch.uint8");
  TORCH_CHECK(mapping.scalar_type() == torch::kInt32,
              "mapping must be torch.int32");
  TORCH_CHECK(x.dim() == 2, "x must have shape [num_samples, input_size]");
  TORCH_CHECK(num_classes >= 2, "num_classes must be >= 2");
  TORCH_CHECK(num_luts > 0 && num_luts % num_classes == 0,
              "num_luts must be positive and divisible by num_classes");
  TORCH_CHECK(binary_luts.dim() == 2 && binary_luts.size(0) == num_luts &&
                  binary_luts.size(1) == lut_size_for_n(fan_in),
              "binary_luts has wrong shape");
  TORCH_CHECK(mapping.dim() == 2 && mapping.size(0) == num_luts &&
                  mapping.size(1) == fan_in,
              "mapping has wrong shape");

  const int32_t* mp = mapping.data_ptr<int32_t>();
  const int64_t slots = num_luts * fan_in;
  for (int64_t i = 0; i < slots; ++i) {
    TORCH_CHECK(mp[i] >= 0 && mp[i] < x.size(1),
                "mapping contains an out-of-range feature index");
  }
}

template <int FixedN>
torch::Tensor predict_impl(
    torch::Tensor x,
    torch::Tensor binary_luts,
    torch::Tensor mapping,
    int runtime_n,
    int64_t num_classes64,
    int64_t num_luts64) {
  const int fan_in = FixedN > 0 ? FixedN : runtime_n;
  check_predict_inputs(x, binary_luts, mapping, fan_in,
                       num_classes64, num_luts64);

  const int num_classes = static_cast<int>(num_classes64);
  const int num_luts = static_cast<int>(num_luts64);
  const int luts_per_class = num_luts / num_classes;
  const int64_t n_samples = x.size(0);
  const int64_t input_size = x.size(1);
  const int64_t lut_size = lut_size_for_n(fan_in);
  const int64_t words_per_lut = (lut_size + 63) / 64;

  const uint8_t* x_ptr = x.data_ptr<uint8_t>();
  const uint8_t* lut_ptr = binary_luts.data_ptr<uint8_t>();
  const int32_t* map_ptr = mapping.data_ptr<int32_t>();

  std::vector<uint64_t> masks(
      static_cast<size_t>(static_cast<int64_t>(num_luts) * words_per_lut), 0ull);
  for (int j = 0; j < num_luts; ++j) {
    for (int64_t a = 0; a < lut_size; ++a) {
      if (lut_ptr[static_cast<int64_t>(j) * lut_size + a] != 0) {
        masks[static_cast<size_t>(static_cast<int64_t>(j) * words_per_lut + (a >> 6))] |=
            uint64_t{1} << (a & 63);
      }
    }
  }

  auto predictions = torch::empty(
      {n_samples},
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* pred_ptr = predictions.data_ptr<int64_t>();

  at::parallel_for(0, n_samples, 256, [&](int64_t begin, int64_t end) {
    std::vector<int> counts(static_cast<size_t>(num_classes));
    for (int64_t b = begin; b < end; ++b) {
      std::fill(counts.begin(), counts.end(), 0);
      const uint8_t* row = x_ptr + b * input_size;

      for (int j = 0; j < num_luts; ++j) {
        uint32_t addr = 0;
        const int32_t* m = map_ptr + static_cast<int64_t>(j) * fan_in;
        for (int bit = 0; bit < fan_in; ++bit) {
          addr |= static_cast<uint32_t>(row[m[bit]] != 0) << bit;
        }
        const uint64_t word = masks[
            static_cast<size_t>(static_cast<int64_t>(j) * words_per_lut + (addr >> 6))];
        const int vote = static_cast<int>((word >> (addr & 63u)) & 1ull);
        counts[static_cast<size_t>(j / luts_per_class)] += vote;
      }

      if (num_classes == 2) {
        pred_ptr[b] = (counts[0] > counts[1]) ? 0 : 1;
      } else {
        int pred = 0;
        int best = counts[0];
        for (int c = 1; c < num_classes; ++c) {
          if (counts[static_cast<size_t>(c)] > best) {
            best = counts[static_cast<size_t>(c)];
            pred = c;
          }
        }
        pred_ptr[b] = pred;
      }
    }
  });

  return predictions;
}


#define DEFINE_FIXED_RANDOM_TRAIN(N) \
std::vector<torch::Tensor> train_fixed_lut##N( \
    torch::Tensor x, torch::Tensor y, torch::Tensor initial_luts, \
    torch::Tensor mapping, float tau, float learning_rate, \
    int64_t epochs, int64_t batch_size, int64_t lr_decay_steps, float gamma, \
    int64_t num_classes, int64_t num_luts, bool verbose) { \
  return train_fixed_impl<N>(x, y, initial_luts, mapping, N, tau, \
                             learning_rate, epochs, batch_size, lr_decay_steps, gamma, \
                             num_classes, num_luts, verbose); \
}

#define DEFINE_FIXED_TRAIN(N) \
std::vector<torch::Tensor> train_lut##N( \
    torch::Tensor x, torch::Tensor y, torch::Tensor initial_luts, \
    torch::Tensor initial_mapping_weights, float tau, float learning_rate, \
    int64_t epochs, int64_t batch_size, int64_t lr_decay_steps, float gamma, \
    int64_t num_classes, int64_t num_luts, bool verbose) { \
  auto out = train_impl<N>(x, y, initial_luts, initial_mapping_weights, N, tau, \
                       learning_rate, epochs, batch_size, lr_decay_steps, gamma, \
                       num_classes, num_luts, 0, verbose); \
  return {out[2], out[3]}; \
}

#define DEFINE_FIXED_TRAIN_STATE(N) \
std::vector<torch::Tensor> train_state_lut##N( \
    torch::Tensor x, torch::Tensor y, torch::Tensor initial_luts, \
    torch::Tensor initial_mapping_weights, float tau, float learning_rate, \
    int64_t epochs, int64_t batch_size, int64_t lr_decay_steps, float gamma, \
    int64_t num_classes, int64_t num_luts, int64_t start_epoch, bool verbose) { \
  return train_impl<N>(x, y, initial_luts, initial_mapping_weights, N, tau, \
                       learning_rate, epochs, batch_size, lr_decay_steps, gamma, \
                       num_classes, num_luts, start_epoch, verbose); \
}

#define DEFINE_FIXED_PREDICT(N) \
torch::Tensor predict_lut##N( \
    torch::Tensor x, torch::Tensor binary_luts, torch::Tensor mapping, \
    int64_t num_classes, int64_t num_luts) { \
  return predict_impl<N>(x, binary_luts, mapping, N, num_classes, num_luts); \
}

DEFINE_FIXED_TRAIN(2)
DEFINE_FIXED_TRAIN(3)
DEFINE_FIXED_TRAIN(4)
DEFINE_FIXED_TRAIN(5)
DEFINE_FIXED_TRAIN(6)
DEFINE_FIXED_TRAIN(8)

DEFINE_FIXED_RANDOM_TRAIN(2)
DEFINE_FIXED_RANDOM_TRAIN(3)
DEFINE_FIXED_RANDOM_TRAIN(4)
DEFINE_FIXED_RANDOM_TRAIN(5)
DEFINE_FIXED_RANDOM_TRAIN(6)
DEFINE_FIXED_RANDOM_TRAIN(8)

DEFINE_FIXED_TRAIN_STATE(2)
DEFINE_FIXED_TRAIN_STATE(3)
DEFINE_FIXED_TRAIN_STATE(4)
DEFINE_FIXED_TRAIN_STATE(5)
DEFINE_FIXED_TRAIN_STATE(6)
DEFINE_FIXED_TRAIN_STATE(8)

DEFINE_FIXED_PREDICT(2)
DEFINE_FIXED_PREDICT(3)
DEFINE_FIXED_PREDICT(4)
DEFINE_FIXED_PREDICT(5)
DEFINE_FIXED_PREDICT(6)
DEFINE_FIXED_PREDICT(8)

std::vector<torch::Tensor> train_general(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor initial_luts,
    torch::Tensor initial_mapping_weights,
    int64_t n,
    float tau,
    float learning_rate,
    int64_t epochs,
    int64_t batch_size,
    int64_t lr_decay_steps,
    float gamma,
    int64_t num_classes,
    int64_t num_luts,
    bool verbose) {
  TORCH_CHECK(n >= 1 && n < 31, "n must be in [1, 30]");
  auto out = train_impl<0>(x, y, initial_luts, initial_mapping_weights,
                       static_cast<int>(n), tau, learning_rate, epochs,
                       batch_size, lr_decay_steps, gamma,
                       num_classes, num_luts, 0, verbose);
  return {out[2], out[3]};
}


std::vector<torch::Tensor> train_fixed_general(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor initial_luts,
    torch::Tensor mapping,
    int64_t n,
    float tau,
    float learning_rate,
    int64_t epochs,
    int64_t batch_size,
    int64_t lr_decay_steps,
    float gamma,
    int64_t num_classes,
    int64_t num_luts,
    bool verbose) {
  TORCH_CHECK(n >= 1 && n < 31, "n must be in [1, 30]");
  return train_fixed_impl<0>(x, y, initial_luts, mapping,
                             static_cast<int>(n), tau, learning_rate, epochs,
                             batch_size, lr_decay_steps, gamma,
                             num_classes, num_luts, verbose);
}

std::vector<torch::Tensor> train_state_general(
    torch::Tensor x,
    torch::Tensor y,
    torch::Tensor initial_luts,
    torch::Tensor initial_mapping_weights,
    int64_t n,
    float tau,
    float learning_rate,
    int64_t epochs,
    int64_t batch_size,
    int64_t lr_decay_steps,
    float gamma,
    int64_t num_classes,
    int64_t num_luts,
    int64_t start_epoch,
    bool verbose) {
  TORCH_CHECK(n >= 1 && n < 31, "n must be in [1, 30]");
  return train_impl<0>(x, y, initial_luts, initial_mapping_weights,
                       static_cast<int>(n), tau, learning_rate, epochs,
                       batch_size, lr_decay_steps, gamma,
                       num_classes, num_luts, start_epoch, verbose);
}

torch::Tensor predict_general(
    torch::Tensor x,
    torch::Tensor binary_luts,
    torch::Tensor mapping,
    int64_t n,
    int64_t num_classes,
    int64_t num_luts) {
  TORCH_CHECK(n >= 1 && n < 31, "n must be in [1, 30]");
  return predict_impl<0>(x, binary_luts, mapping, static_cast<int>(n),
                         num_classes, num_luts);
}

void bind_train(py::module_& m, const char* name,
                std::vector<torch::Tensor> (*fn)(torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, float, float, int64_t,
                    int64_t, int64_t, float, int64_t, int64_t, bool)) {
  m.def(name, fn,
        py::arg("x"), py::arg("y"), py::arg("initial_luts"),
        py::arg("initial_mapping_weights"), py::arg("tau"),
        py::arg("learning_rate"), py::arg("epochs"),
        py::arg("batch_size") = 512,
        py::arg("lr_decay_steps") = 8,
        py::arg("gamma") = 0.1f,
        py::arg("num_classes") = 2,
        py::arg("num_luts") = 40,
        py::arg("verbose") = true);
}


void bind_train_fixed(py::module_& m, const char* name,
                std::vector<torch::Tensor> (*fn)(torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, float, float, int64_t,
                    int64_t, int64_t, float, int64_t, int64_t, bool)) {
  m.def(name, fn,
        py::arg("x"), py::arg("y"), py::arg("initial_luts"),
        py::arg("mapping"), py::arg("tau"),
        py::arg("learning_rate"), py::arg("epochs"),
        py::arg("batch_size") = 512,
        py::arg("lr_decay_steps") = 8,
        py::arg("gamma") = 0.1f,
        py::arg("num_classes") = 2,
        py::arg("num_luts") = 40,
        py::arg("verbose") = true);
}

void bind_train_state(py::module_& m, const char* name,
                std::vector<torch::Tensor> (*fn)(torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, float, float, int64_t,
                    int64_t, int64_t, float, int64_t, int64_t, int64_t, bool)) {
  m.def(name, fn,
        py::arg("x"), py::arg("y"), py::arg("initial_luts"),
        py::arg("initial_mapping_weights"), py::arg("tau"),
        py::arg("learning_rate"), py::arg("epochs"),
        py::arg("batch_size") = 512,
        py::arg("lr_decay_steps") = 8,
        py::arg("gamma") = 0.1f,
        py::arg("num_classes") = 2,
        py::arg("num_luts") = 40,
        py::arg("start_epoch") = 0,
        py::arg("verbose") = true);
}

void bind_predict(py::module_& m, const char* name,
                  torch::Tensor (*fn)(torch::Tensor, torch::Tensor,
                                      torch::Tensor, int64_t, int64_t)) {
  m.def(name, fn,
        py::arg("x"), py::arg("binary_luts"), py::arg("mapping"),
        py::arg("num_classes") = 2,
        py::arg("num_luts") = 40);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  bind_train(m, "train_lut2", &train_lut2);
  bind_train(m, "train_lut3", &train_lut3);
  bind_train(m, "train_lut4", &train_lut4);
  bind_train(m, "train_lut5", &train_lut5);
  bind_train(m, "train_lut6", &train_lut6);
  bind_train(m, "train_lut8", &train_lut8);

  bind_train_fixed(m, "train_fixed_lut2", &train_fixed_lut2);
  bind_train_fixed(m, "train_fixed_lut3", &train_fixed_lut3);
  bind_train_fixed(m, "train_fixed_lut4", &train_fixed_lut4);
  bind_train_fixed(m, "train_fixed_lut5", &train_fixed_lut5);
  bind_train_fixed(m, "train_fixed_lut6", &train_fixed_lut6);
  bind_train_fixed(m, "train_fixed_lut8", &train_fixed_lut8);

  bind_train_state(m, "train_state_lut2", &train_state_lut2);
  bind_train_state(m, "train_state_lut3", &train_state_lut3);
  bind_train_state(m, "train_state_lut4", &train_state_lut4);
  bind_train_state(m, "train_state_lut5", &train_state_lut5);
  bind_train_state(m, "train_state_lut6", &train_state_lut6);
  bind_train_state(m, "train_state_lut8", &train_state_lut8);

  bind_predict(m, "predict_lut2", &predict_lut2);
  bind_predict(m, "predict_lut3", &predict_lut3);
  bind_predict(m, "predict_lut4", &predict_lut4);
  bind_predict(m, "predict_lut5", &predict_lut5);
  bind_predict(m, "predict_lut6", &predict_lut6);
  bind_predict(m, "predict_lut8", &predict_lut8);

  m.def("train", &train_general,
        py::arg("x"), py::arg("y"), py::arg("initial_luts"),
        py::arg("initial_mapping_weights"), py::arg("n"), py::arg("tau"),
        py::arg("learning_rate"), py::arg("epochs"),
        py::arg("batch_size") = 512,
        py::arg("lr_decay_steps") = 8,
        py::arg("gamma") = 0.1f,
        py::arg("num_classes") = 2,
        py::arg("num_luts") = 40,
        py::arg("verbose") = true);

  m.def("train_general", &train_general,
        py::arg("x"), py::arg("y"), py::arg("initial_luts"),
        py::arg("initial_mapping_weights"), py::arg("n"), py::arg("tau"),
        py::arg("learning_rate"), py::arg("epochs"),
        py::arg("batch_size") = 512,
        py::arg("lr_decay_steps") = 8,
        py::arg("gamma") = 0.1f,
        py::arg("num_classes") = 2,
        py::arg("num_luts") = 40,
        py::arg("verbose") = true);

  m.def("train_fixed", &train_fixed_general,
        py::arg("x"), py::arg("y"), py::arg("initial_luts"),
        py::arg("mapping"), py::arg("n"), py::arg("tau"),
        py::arg("learning_rate"), py::arg("epochs"),
        py::arg("batch_size") = 512,
        py::arg("lr_decay_steps") = 8,
        py::arg("gamma") = 0.1f,
        py::arg("num_classes") = 2,
        py::arg("num_luts") = 40,
        py::arg("verbose") = true);

  m.def("train_fixed_general", &train_fixed_general,
        py::arg("x"), py::arg("y"), py::arg("initial_luts"),
        py::arg("mapping"), py::arg("n"), py::arg("tau"),
        py::arg("learning_rate"), py::arg("epochs"),
        py::arg("batch_size") = 512,
        py::arg("lr_decay_steps") = 8,
        py::arg("gamma") = 0.1f,
        py::arg("num_classes") = 2,
        py::arg("num_luts") = 40,
        py::arg("verbose") = true);

  m.def("train_state", &train_state_general,
        py::arg("x"), py::arg("y"), py::arg("initial_luts"),
        py::arg("initial_mapping_weights"), py::arg("n"), py::arg("tau"),
        py::arg("learning_rate"), py::arg("epochs"),
        py::arg("batch_size") = 512,
        py::arg("lr_decay_steps") = 8,
        py::arg("gamma") = 0.1f,
        py::arg("num_classes") = 2,
        py::arg("num_luts") = 40,
        py::arg("start_epoch") = 0,
        py::arg("verbose") = true);

  m.def("train_state_general", &train_state_general,
        py::arg("x"), py::arg("y"), py::arg("initial_luts"),
        py::arg("initial_mapping_weights"), py::arg("n"), py::arg("tau"),
        py::arg("learning_rate"), py::arg("epochs"),
        py::arg("batch_size") = 512,
        py::arg("lr_decay_steps") = 8,
        py::arg("gamma") = 0.1f,
        py::arg("num_classes") = 2,
        py::arg("num_luts") = 40,
        py::arg("start_epoch") = 0,
        py::arg("verbose") = true);

  m.def("predict", &predict_general,
        py::arg("x"), py::arg("binary_luts"), py::arg("mapping"),
        py::arg("n"), py::arg("num_classes") = 2,
        py::arg("num_luts") = 40);

  m.def("predict_general", &predict_general,
        py::arg("x"), py::arg("binary_luts"), py::arg("mapping"),
        py::arg("n"), py::arg("num_classes") = 2,
        py::arg("num_luts") = 40);
}
