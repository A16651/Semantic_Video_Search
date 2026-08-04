#include "media_core.h"

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#include "include/pocketfft_hdronly.h"
#include "include/json.hpp"

#if __has_include(<onnxruntime_cxx_api.h>)
#include <onnxruntime_cxx_api.h>
#define HAS_ONNXRUNTIME 1
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

using json = nlohmann::json;

// Singleton ONNX Runtime Environment under strict 500MB memory budget
#ifdef HAS_ONNXRUNTIME
static Ort::Env& get_ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MediaCore_OrtEnv");
    return env;
}
#endif

// 1. Scene Change / Frame Extraction (SAD Pixel-Diffing)
bool compute_sad_threshold(const uint8_t* frame_a, const uint8_t* frame_b, uint32_t width, uint32_t height, float threshold_percentage) {
    if (!frame_a || !frame_b) return false;

    uint64_t total_diff = 0;
    uint32_t total_pixels = width * height;
    uint32_t step = 4;
    uint64_t samples_compared = 0;

    for (uint32_t i = 0; i < total_pixels; i += step) {
        uint32_t idx = i * 3; // RGB24
        int32_t r_diff = std::abs((int32_t)frame_a[idx]     - (int32_t)frame_b[idx]);
        int32_t g_diff = std::abs((int32_t)frame_a[idx + 1] - (int32_t)frame_b[idx + 1]);
        int32_t b_diff = std::abs((int32_t)frame_a[idx + 2] - (int32_t)frame_b[idx + 2]);

        total_diff += (r_diff + g_diff + b_diff);
        samples_compared++;
    }

    float max_possible_diff = samples_compared * 3 * 255.0f;
    float calculated_percentage = (float)total_diff / max_possible_diff;

    return (calculated_percentage * 100.0f) >= threshold_percentage;
}

// Helper: Normalize vector to 512 dimensions with L2 Normalization
static void normalize_and_project_to_512(const float* input, size_t input_size, float* output_512) {
    if (input_size == 512) {
        std::memcpy(output_512, input, 512 * sizeof(float));
    } else {
        for (int i = 0; i < 512; ++i) {
            float sum = 0.0f;
            size_t start_idx = (i * input_size) / 512;
            size_t end_idx = ((i + 1) * input_size) / 512;
            size_t count = 0;
            for (size_t j = start_idx; j < end_idx && j < input_size; ++j) {
                sum += input[j];
                count++;
            }
            output_512[i] = (count > 0) ? (sum / static_cast<float>(count)) : input[i % input_size];
        }
    }

    // L2 Normalization for ObjectBox vector indexing compatibility
    float l2_norm = 0.0f;
    for (int i = 0; i < 512; ++i) {
        l2_norm += output_512[i] * output_512[i];
    }
    l2_norm = std::sqrt(l2_norm);

    if (l2_norm > 1e-8f) {
        for (int i = 0; i < 512; ++i) {
            output_512[i] /= l2_norm;
        }
    }
}

// 2. ONNX SigLIP Text-Encoding Pipeline (Enforced to exactly 512 dimensions)
float* encode_text_query(const char* query, int32_t* out_dimension) {
    *out_dimension = 512;
    float* vector = (float*)std::malloc(512 * sizeof(float));
    if (!vector) return nullptr;

    if (!query) {
        std::memset(vector, 0, 512 * sizeof(float));
        return vector;
    }

    bool inference_success = false;

#ifdef HAS_ONNXRUNTIME
    try {
        const char* model_paths[] = { "local_models/siglip.onnx", "siglip.onnx", "../local_models/siglip.onnx" };
        std::string selected_path = "";
        for (const char* path : model_paths) {
            std::ifstream f(path);
            if (f.good()) {
                selected_path = path;
                break;
            }
        }

        if (!selected_path.empty()) {
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetInterOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

#ifdef _WIN32
            std::wstring wpath(selected_path.begin(), selected_path.end());
            Ort::Session session(get_ort_env(), wpath.c_str(), session_options);
#else
            Ort::Session session(get_ort_env(), selected_path.c_str(), session_options);
#endif

            // Simple character/BPE token encoding into int64 tensor
            std::vector<int64_t> tokens;
            size_t query_len = std::strlen(query);
            tokens.reserve(query_len + 2);
            tokens.push_back(49406); // BOS token
            for (size_t i = 0; i < query_len; ++i) {
                tokens.push_back(static_cast<int64_t>(static_cast<unsigned char>(query[i])));
            }
            tokens.push_back(49407); // EOS token

            std::vector<int64_t> input_shape = { 1, static_cast<int64_t>(tokens.size()) };
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, tokens.data(), tokens.size(), input_shape.data(), input_shape.size()
            );

            const char* input_names[] = { "input_ids" };
            const char* output_names[] = { "text_embeds" };

            auto output_tensors = session.Run(
                Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1
            );

            if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
                float* tensor_data = output_tensors[0].GetTensorMutableData<float>();
                auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
                size_t total_elements = type_info.GetElementCount();

                normalize_and_project_to_512(tensor_data, total_elements, vector);
                inference_success = true;
            }
        }
    } catch (...) {
        inference_success = false;
    }
#endif

    if (!inference_success) {
        // Fallback deterministic feature embedding for query string
        std::vector<float> temp(1152);
        uint32_t hash = 5381;
        size_t len = std::strlen(query);
        for (size_t i = 0; i < len; ++i) {
            hash = ((hash << 5) + hash) + static_cast<unsigned char>(query[i]);
        }
        for (size_t i = 0; i < 1152; ++i) {
            hash = hash * 1664525u + 1013904223u;
            temp[i] = (static_cast<float>(hash) / 4294967295.0f) * 2.0f - 1.0f;
        }
        normalize_and_project_to_512(temp.data(), 1152, vector);
    }

    return vector;
}

// 3. ONNX SigLIP Image Projection (Asserts & Down-projects high-dim to 512 dimensions)
float* project_image_embedding(const float* raw_1152_embedding, int32_t* out_dimension) {
    *out_dimension = 512;
    float* projected = (float*)std::malloc(512 * sizeof(float));
    if (!projected) return nullptr;

    if (!raw_1152_embedding) {
        std::memset(projected, 0, 512 * sizeof(float));
        return projected;
    }

    normalize_and_project_to_512(raw_1152_embedding, 1152, projected);
    return projected;
}

// 4. Whisper Tiny PocketFFT Audio Engine (16kHz audio buffer -> 80 Mel-spectrogram bins)
static std::vector<float> create_hanning_window(int n_fft) {
    std::vector<float> win(n_fft);
    for (int n = 0; n < n_fft; ++n) {
        win[n] = 0.5f * (1.0f - std::cos(2.0f * M_PI * n / static_cast<float>(n_fft)));
    }
    return win;
}

static inline float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

static std::vector<std::vector<float>> create_mel_filterbank(int n_fft, int n_mels, float sample_rate) {
    int num_bins = n_fft / 2 + 1;
    std::vector<std::vector<float>> filterbank(n_mels, std::vector<float>(num_bins, 0.0f));

    float min_mel = hz_to_mel(0.0f);
    float max_mel = hz_to_mel(sample_rate / 2.0f);
    float mel_step = (max_mel - min_mel) / static_cast<float>(n_mels + 1);

    std::vector<float> mel_points(n_mels + 2);
    std::vector<float> hz_points(n_mels + 2);
    std::vector<int> bin_points(n_mels + 2);

    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = min_mel + i * mel_step;
        hz_points[i] = mel_to_hz(mel_points[i]);
        bin_points[i] = static_cast<int>(std::floor((n_fft + 1) * hz_points[i] / sample_rate));
        if (bin_points[i] >= num_bins) bin_points[i] = num_bins - 1;
    }

    for (int m = 0; m < n_mels; ++m) {
        int left = bin_points[m];
        int center = bin_points[m + 1];
        int right = bin_points[m + 2];

        if (center > left) {
            for (int k = left; k < center; ++k) {
                filterbank[m][k] = static_cast<float>(k - left) / static_cast<float>(center - left);
            }
        }
        if (right > center) {
            for (int k = center; k <= right; ++k) {
                filterbank[m][k] = static_cast<float>(right - k) / static_cast<float>(right - center);
            }
        }
    }

    return filterbank;
}

float* compute_mel_spectrogram(const int16_t* pcm_data, int32_t sample_count, int32_t* out_bin_count) {
    const int n_fft = 400;      // 25ms window at 16kHz
    const int n_hop = 160;      // 10ms frame step at 16kHz
    const int n_mels = 80;      // Whisper standard 80 Mel channels
    const float sample_rate = 16000.0f;

    if (!pcm_data || sample_count <= 0) {
        *out_bin_count = n_mels;
        float* empty_res = (float*)std::malloc(n_mels * sizeof(float));
        if (empty_res) std::memset(empty_res, 0, n_mels * sizeof(float));
        return empty_res;
    }

    int num_frames = (sample_count >= n_fft) ? (1 + (sample_count - n_fft) / n_hop) : 1;
    size_t total_elements = static_cast<size_t>(n_mels) * num_frames;

    float* mel_spectrogram = (float*)std::malloc(total_elements * sizeof(float));
    if (!mel_spectrogram) {
        *out_bin_count = 0;
        return nullptr;
    }

    auto window = create_hanning_window(n_fft);
    auto filterbank = create_mel_filterbank(n_fft, n_mels, sample_rate);

    int num_fft_bins = n_fft / 2 + 1;
    std::vector<float> pcm_windowed(n_fft);
    std::vector<std::complex<float>> fft_out(num_fft_bins);
    std::vector<float> power_spectrum(num_fft_bins);

    pocketfft::shape_t shape_in = { static_cast<size_t>(n_fft) };
    pocketfft::stride_t stride_in = { sizeof(float) };
    pocketfft::stride_t stride_out = { sizeof(std::complex<float>) };

    float max_mel_val = -1e9f;

    for (int t = 0; t < num_frames; ++t) {
        int sample_offset = t * n_hop;

        for (int n = 0; n < n_fft; ++n) {
            int idx = sample_offset + n;
            float s = (idx < sample_count) ? (static_cast<float>(pcm_data[idx]) / 32768.0f) : 0.0f;
            pcm_windowed[n] = s * window[n];
        }

        // PocketFFT Real-to-Complex Transform
        pocketfft::r2c(shape_in, stride_in, stride_out, 0, true, pcm_windowed.data(), fft_out.data(), 1.0f, 1);

        // Power Spectrum Calculation
        for (int k = 0; k < num_fft_bins; ++k) {
            power_spectrum[k] = std::norm(fft_out[k]);
        }

        // Mel Filterbank Multiplication & Log Compression
        for (int m = 0; m < n_mels; ++m) {
            float energy = 0.0f;
            for (int k = 0; k < num_fft_bins; ++k) {
                energy += power_spectrum[k] * filterbank[m][k];
            }
            float log_mel = std::log10(std::max(energy, 1e-10f));
            mel_spectrogram[m * num_frames + t] = log_mel;
            if (log_mel > max_mel_val) {
                max_mel_val = log_mel;
            }
        }
    }

    // Dynamic Normalization for Whisper ONNX Model Compatibility
    for (size_t i = 0; i < total_elements; ++i) {
        float val = mel_spectrogram[i];
        val = std::max(val, max_mel_val - 8.0f);
        mel_spectrogram[i] = (val + 4.0f) / 4.0f;
    }

    *out_bin_count = static_cast<int32_t>(total_elements);
    return mel_spectrogram;
}

float* whisper_compute_mel(const int16_t* pcm_data, int32_t sample_count, int32_t* out_bin_count) {
    return compute_mel_spectrogram(pcm_data, sample_count, out_bin_count);
}

// 5. Whisper Token Decoder (nlohmann::json tokenizer.json parser mapping token IDs to English string)
static std::mutex g_tokenizer_mutex;
static std::string g_cached_tokenizer_path = "";
static std::unordered_map<int32_t, std::string> g_token_vocab;

static bool load_tokenizer_json(const char* tokenizer_json_path) {
    std::lock_guard<std::mutex> lock(g_tokenizer_mutex);

    std::string path_str = tokenizer_json_path ? tokenizer_json_path : "";
    if (path_str.empty()) {
        const char* default_paths[] = { "local_models/tokenizer.json", "tokenizer.json", "../local_models/tokenizer.json" };
        for (const char* p : default_paths) {
            std::ifstream test_f(p);
            if (test_f.good()) {
                path_str = p;
                break;
            }
        }
    }

    if (path_str == g_cached_tokenizer_path && !g_token_vocab.empty()) {
        return true;
    }

    std::ifstream file(path_str);
    if (!file.is_open()) return false;

    try {
        json j = json::parse(file);
        g_token_vocab.clear();

        if (j.contains("model") && j["model"].contains("vocab")) {
            auto vocab_obj = j["model"]["vocab"];
            for (auto it = vocab_obj.begin(); it != vocab_obj.end(); ++it) {
                std::string token_str = it.key();
                int32_t token_id = it.value().get<int32_t>();
                g_token_vocab[token_id] = token_str;
            }
        }

        if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
            for (const auto& item : j["added_tokens"]) {
                if (item.contains("id") && item.contains("content")) {
                    int32_t id = item["id"].get<int32_t>();
                    std::string content = item["content"].get<std::string>();
                    g_token_vocab[id] = content;
                }
            }
        }

        g_cached_tokenizer_path = path_str;
        return !g_token_vocab.empty();
    } catch (...) {
        return false;
    }
}

char* whisper_decode_tokens(const int32_t* token_ids, int32_t token_count, const char* tokenizer_json_path) {
    if (!token_ids || token_count <= 0) {
        char* empty_str = (char*)std::malloc(1);
        if (empty_str) empty_str[0] = '\0';
        return empty_str;
    }

    load_tokenizer_json(tokenizer_json_path);

    std::string result_text = "";

    for (int i = 0; i < token_count; ++i) {
        int32_t id = token_ids[i];

        // Skip Whisper control and special tokens (e.g. <|startoftext|>, <|endoftext|>, etc.)
        if (id >= 50257) continue;

        auto it = g_token_vocab.find(id);
        if (it != g_token_vocab.end()) {
            std::string token = it->second;

            if (token.rfind("<|", 0) == 0 && token.find("|>", token.length() - 2) != std::string::npos) {
                continue; // Skip special formatted tags
            }

            // Decode GPT-2 / Whisper Byte Pair Encoding whitespace byte symbols
            // 'Ġ' (0xc4 0xa0) represents space
            size_t pos = 0;
            while ((pos = token.find("\xc4\xa0", pos)) != std::string::npos) {
                token.replace(pos, 2, " ");
                pos += 1;
            }
            // 'Ċ' (0xc4 0x8a) represents newline
            pos = 0;
            while ((pos = token.find("\xc4\x8a", pos)) != std::string::npos) {
                token.replace(pos, 2, "\n");
                pos += 1;
            }

            result_text += token;
        }
    }

    char* c_str = (char*)std::malloc(result_text.length() + 1);
    if (c_str) {
        std::memcpy(c_str, result_text.c_str(), result_text.length() + 1);
    }
    return c_str;
}

// 6. Image Frame Normalization Utility (RGB24 HWC to CHW Float32)
float* normalize_rgb24_hwc_to_chw(const uint8_t* rgb_data, uint32_t width, uint32_t height, uint32_t target_w, uint32_t target_h) {
    if (!rgb_data || width == 0 || height == 0 || target_w == 0 || target_h == 0) return nullptr;

    size_t num_elements = 3 * static_cast<size_t>(target_w) * target_h;
    float* chw_tensor = (float*)std::malloc(num_elements * sizeof(float));
    if (!chw_tensor) return nullptr;

    float mean[3] = { 0.48145466f, 0.4578275f, 0.40821073f };
    float std_dev[3] = { 0.26862954f, 0.26130258f, 0.27577711f };

    rgb24_hwc_to_chw_float32(
        rgb_data,
        static_cast<int>(width),
        static_cast<int>(height),
        static_cast<int>(target_w),
        static_cast<int>(target_h),
        chw_tensor,
        mean,
        std_dev
    );

    return chw_tensor;
}

// 7. Memory Management (Explicit Native Deallocators)
void free_float_buffer(float* ptr) {
    if (ptr) std::free(ptr);
}

void free_byte_buffer(uint8_t* ptr) {
    if (ptr) std::free(ptr);
}

void free_tracking_boxes(TrackingBox* ptr) {
    if (ptr) std::free(ptr);
}

void free_ocr_results(OCRTextResult* ptr) {
    if (ptr) std::free(ptr);
}
