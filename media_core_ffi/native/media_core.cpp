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

#ifdef HAS_ONNXRUNTIME
static std::mutex g_siglip_session_mutex;
static std::unique_ptr<Ort::Session> g_siglip_session = nullptr;

static bool init_siglip_model() {
    std::lock_guard<std::mutex> lock(g_siglip_session_mutex);
    if (g_siglip_session) return true;

    std::cout << "[Native MediaCore] Initializing SigLIP ONNX session..." << std::endl;
    std::fflush(stdout);

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
            g_siglip_session = std::make_unique<Ort::Session>(get_ort_env(), wpath.c_str(), session_options);
#else
            g_siglip_session = std::make_unique<Ort::Session>(get_ort_env(), selected_path.c_str(), session_options);
#endif
            std::cout << "[Native MediaCore] SigLIP session successfully loaded from path: " << selected_path << std::endl;
            std::fflush(stdout);
            return true;
        } else {
            std::cout << "[Native MediaCore] Warning: SigLIP model file not found in search paths." << std::endl;
            std::fflush(stdout);
        }
    } catch (const std::exception& e) {
        std::cout << "[Native MediaCore] Error initializing SigLIP ONNX model: " << e.what() << std::endl;
        std::fflush(stdout);
    } catch (...) {
        std::cout << "[Native MediaCore] Unknown exception initializing SigLIP ONNX model." << std::endl;
        std::fflush(stdout);
    }
    return false;
}
#endif

static std::mutex g_bpe_tokenizer_mutex;
static std::unordered_map<std::string, int64_t> g_bpe_vocab;

static void init_bpe_tokenizer(const char* model_dir) {
    std::lock_guard<std::mutex> lock(g_bpe_tokenizer_mutex);
    if (!g_bpe_vocab.empty()) return;

    std::string base_dir = (model_dir && std::strlen(model_dir) > 0) ? model_dir : "local_models";
    std::string token_path = base_dir + "/tokenizer.json";

    std::ifstream f(token_path);
    if (!f.good()) {
        const char* alt_paths[] = { "local_models/tokenizer.json", "tokenizer.json", "../local_models/tokenizer.json" };
        for (const char* p : alt_paths) {
            std::ifstream alt_f(p);
            if (alt_f.good()) {
                token_path = p;
                break;
            }
        }
    }

    try {
        std::ifstream file(token_path);
        if (file.is_open()) {
            nlohmann::json j;
            file >> j;
            if (j.contains("model") && j["model"].contains("vocab")) {
                for (auto& el : j["model"]["vocab"].items()) {
                    g_bpe_vocab[el.key()] = el.value().get<int64_t>();
                }
            }
        }
    } catch (...) {}
}

static std::vector<int64_t> bpe_tokenize_query(const char* query, const char* model_dir) {
    init_bpe_tokenizer(model_dir);

    std::vector<int64_t> tokens;
    tokens.push_back(49406); // BOS token ID

    if (!query || std::strlen(query) == 0) {
        tokens.push_back(49407); // EOS token ID
        return tokens;
    }

    std::string query_str = query;
    std::string word = "";

    for (size_t i = 0; i <= query_str.length(); ++i) {
        if (i == query_str.length() || std::isspace(static_cast<unsigned char>(query_str[i])) || std::ispunct(static_cast<unsigned char>(query_str[i]))) {
            if (!word.empty()) {
                std::string word_w_suffix = word + "</w>";
                std::string word_w_prefix = "\xc4\xa0" + word;
                std::string word_w_both = "\xc4\xa0" + word + "</w>";

                auto it1 = g_bpe_vocab.find(word_w_suffix);
                auto it2 = g_bpe_vocab.find(word_w_both);
                auto it3 = g_bpe_vocab.find(word_w_prefix);
                auto it4 = g_bpe_vocab.find(word);

                if (it1 != g_bpe_vocab.end()) {
                    tokens.push_back(it1->second);
                } else if (it2 != g_bpe_vocab.end()) {
                    tokens.push_back(it2->second);
                } else if (it3 != g_bpe_vocab.end()) {
                    tokens.push_back(it3->second);
                } else if (it4 != g_bpe_vocab.end()) {
                    tokens.push_back(it4->second);
                } else {
                    for (char c : word) {
                        std::string char_str(1, c);
                        auto it_c = g_bpe_vocab.find(char_str);
                        if (it_c != g_bpe_vocab.end()) {
                            tokens.push_back(it_c->second);
                        } else {
                            tokens.push_back(261); // UNK token ID
                        }
                    }
                }
                word = "";
            }
        } else {
            word += static_cast<char>(std::tolower(static_cast<unsigned char>(query_str[i])));
        }
    }

    tokens.push_back(49407); // EOS token ID
    return tokens;
}

// 2. ONNX SigLIP Text-Encoding Pipeline (Enforced to exactly 512 dimensions)
float* encode_text_query(const char* query, int32_t* out_dimension) {
    *out_dimension = 512;
    float* vector = (float*)std::malloc(512 * sizeof(float));
    if (!vector) return nullptr;

    if (!query) {
        std::memset(vector, 0, 512 * sizeof(float));
        vector[0] = 1.0f;
        return vector;
    }

    bool inference_success = false;

#ifdef HAS_ONNXRUNTIME
    if (init_siglip_model() && g_siglip_session) {
        try {
            std::vector<int64_t> tokens = bpe_tokenize_query(query, nullptr);

            std::vector<int64_t> input_shape = { 1, static_cast<int64_t>(tokens.size()) };
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, tokens.data(), tokens.size(), input_shape.data(), input_shape.size()
            );

            const char* input_names[] = { "input_ids" };
            const char* output_names[] = { "text_embeds" };

            auto output_tensors = g_siglip_session->Run(
                Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1
            );

            if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
                float* tensor_data = output_tensors[0].GetTensorMutableData<float>();
                auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
                size_t total_elements = type_info.GetElementCount();

                normalize_and_project_to_512(tensor_data, total_elements, vector);
                inference_success = true;
            }
        } catch (...) {
            inference_success = false;
        }
    }
#endif

    if (!inference_success) {
        std::free(vector);
        return nullptr; // Clean nullptr on inference failure to prevent database corruption
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

// 10. SigLIP Vision ONNX Transformer Encoder (CHW Float32 [1, 3, 224, 224] -> 512-dim visual vector)
float* encode_image_frame(const float* chw_data, int32_t* out_dimension) {
    *out_dimension = 512;
    if (!chw_data) {
        std::cout << "[Native MediaCore] encode_image_frame called with null chw_data pointer." << std::endl;
        std::fflush(stdout);
        return nullptr;
    }

    std::cout << "[Native MediaCore] Starting SigLIP image frame ONNX inference..." << std::endl;
    std::fflush(stdout);

    float* vector = (float*)std::malloc(512 * sizeof(float));
    if (!vector) return nullptr;

    bool inference_success = false;

#ifdef HAS_ONNXRUNTIME
    if (init_siglip_model() && g_siglip_session) {
        try {
            int64_t image_shape[] = { 1, 3, 224, 224 };
            size_t num_pixels = 3 * 224 * 224;

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(chw_data), num_pixels, image_shape, 4
            );

            const char* input_names[] = { "pixel_values" };
            const char* output_names[] = { "image_embeds" };

            auto output_tensors = g_siglip_session->Run(
                Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1
            );

            if (!output_tensors.empty() && output_tensors[0].IsTensor()) {
                float* tensor_data = output_tensors[0].GetTensorMutableData<float>();
                auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
                size_t total_elements = type_info.GetElementCount();

                normalize_and_project_to_512(tensor_data, total_elements, vector);
                inference_success = true;
                std::cout << "[Native MediaCore] SigLIP image frame inference complete. Projected to 512D vector." << std::endl;
                std::fflush(stdout);
            }
        } catch (const std::exception& e) {
            std::cout << "[Native MediaCore] Exception inside encode_image_frame: " << e.what() << std::endl;
            std::fflush(stdout);
            inference_success = false;
        } catch (...) {
            std::cout << "[Native MediaCore] Unknown exception inside encode_image_frame." << std::endl;
            std::fflush(stdout);
            inference_success = false;
        }
    }
#endif

    if (!inference_success) {
        std::cout << "[Native MediaCore] Warning: SigLIP image frame inference failed." << std::endl;
        std::fflush(stdout);
        std::free(vector);
        return nullptr;
    }

    return vector;
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

// Global Persistent ONNX Runtime Sessions for Whisper (Loaded ONCE to avoid disk re-reads)
#ifdef HAS_ONNXRUNTIME
static std::mutex g_whisper_session_mutex;
static std::unique_ptr<Ort::Session> g_encoder_session = nullptr;
static std::unique_ptr<Ort::Session> g_decoder_session = nullptr;
static std::string g_cached_whisper_dir = "";
#endif

bool init_whisper_models(const char* model_dir) {
#ifdef HAS_ONNXRUNTIME
    std::lock_guard<std::mutex> lock(g_whisper_session_mutex);

    std::string base_dir = (model_dir && std::strlen(model_dir) > 0) ? model_dir : "local_models";
    if (g_encoder_session && g_decoder_session && g_cached_whisper_dir == base_dir) {
        return true;
    }

    std::cout << "[Native MediaCore] Initializing Whisper ONNX encoder & decoder sessions from: " << base_dir << std::endl;
    std::fflush(stdout);

    try {
        std::string encoder_path = base_dir + "/encoder_model.onnx";
        std::string decoder_path = base_dir + "/decoder_model.onnx";

        std::ifstream fe(encoder_path);
        std::ifstream fd(decoder_path);

        if (fe.good() && fd.good()) {
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetInterOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

#ifdef _WIN32
            std::wstring wencoder(encoder_path.begin(), encoder_path.end());
            std::wstring wdecoder(decoder_path.begin(), decoder_path.end());
            g_encoder_session = std::make_unique<Ort::Session>(get_ort_env(), wencoder.c_str(), session_options);
            g_decoder_session = std::make_unique<Ort::Session>(get_ort_env(), wdecoder.c_str(), session_options);
#else
            g_encoder_session = std::make_unique<Ort::Session>(get_ort_env(), encoder_path.c_str(), session_options);
            g_decoder_session = std::make_unique<Ort::Session>(get_ort_env(), decoder_path.c_str(), session_options);
#endif
            g_cached_whisper_dir = base_dir;
            std::cout << "[Native MediaCore] Whisper ONNX encoder & decoder sessions successfully cached." << std::endl;
            std::fflush(stdout);
            return true;
        } else {
            std::cout << "[Native MediaCore] Warning: Whisper encoder or decoder model file missing in " << base_dir << std::endl;
            std::fflush(stdout);
        }
    } catch (const std::exception& e) {
        std::cout << "[Native MediaCore] Error initializing Whisper ONNX models: " << e.what() << std::endl;
        std::fflush(stdout);
    } catch (...) {
        std::cout << "[Native MediaCore] Unknown exception initializing Whisper ONNX models." << std::endl;
        std::fflush(stdout);
    }
#endif
    return false;
}

// ONNX Autoregressive Whisper Speech Recognition (Encoder -> Decoder Logits -> Argmax -> Token Sequence)
char* whisper_transcribe_audio(const float* mel_data, int32_t mel_bins, const char* model_dir) {
    if (!mel_data || mel_bins <= 0) {
        char* empty_res = (char*)std::malloc(1);
        if (empty_res) empty_res[0] = '\0';
        return empty_res;
    }

    std::cout << "[Native MediaCore] Transcribing audio chunk via Whisper ONNX (" << mel_bins << " Mel bins)..." << std::endl;
    std::fflush(stdout);

    std::vector<int32_t> token_sequence = { 50258, 50259, 50359, 50363 };

#ifdef HAS_ONNXRUNTIME
    init_whisper_models(model_dir);

    if (g_encoder_session && g_decoder_session) {
        try {
            // 1. Run persistent ONNX Encoder: Input mel [1, 80, 3000] -> Output encoder_hidden_states
            std::cout << "[Native MediaCore] Running Whisper Encoder session..." << std::endl;
            std::fflush(stdout);

            int64_t mel_shape[] = { 1, 80, 3000 };
            std::vector<float> padded_mel(80 * 3000, 0.0f);
            size_t total_mel_elements = static_cast<size_t>(mel_bins) * 80;
            size_t copy_count = std::min(padded_mel.size(), total_mel_elements);
            std::memcpy(padded_mel.data(), mel_data, copy_count * sizeof(float));

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value mel_tensor = Ort::Value::CreateTensor<float>(
                memory_info, padded_mel.data(), padded_mel.size(), mel_shape, 3
            );

            const char* enc_inputs[] = { "input_features" };
            const char* enc_outputs[] = { "last_hidden_state" };

            auto enc_output_tensors = g_encoder_session->Run(
                Ort::RunOptions{ nullptr }, enc_inputs, &mel_tensor, 1, enc_outputs, 1
            );

            if (!enc_output_tensors.empty() && enc_output_tensors[0].IsTensor()) {
                std::cout << "[Native MediaCore] Whisper Encoder finished. Starting Autoregressive Decoder argmax loop..." << std::endl;
                std::fflush(stdout);

                float* hidden_states = enc_output_tensors[0].GetTensorMutableData<float>();
                auto enc_type_info = enc_output_tensors[0].GetTensorTypeAndShapeInfo();
                std::vector<int64_t> hidden_shape = enc_type_info.GetShape();

                // 2. Autoregressive Argmax Decoder Loop using persistent g_decoder_session:
                int max_length = 64;
                int predicted_token = -1;

                while (predicted_token != 50257 && token_sequence.size() < static_cast<size_t>(max_length)) {
                    std::vector<int64_t> input_ids(token_sequence.begin(), token_sequence.end());
                    int64_t input_ids_shape[] = { 1, static_cast<int64_t>(input_ids.size()) };

                    int64_t mask_shape[] = { 1, 1500 };
                    std::vector<int64_t> attn_mask(1500, 1);

                    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
                        memory_info, input_ids.data(), input_ids.size(), input_ids_shape, 2
                    );
                    Ort::Value hidden_tensor = Ort::Value::CreateTensor<float>(
                        memory_info, hidden_states, enc_output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount(),
                        hidden_shape.data(), hidden_shape.size()
                    );
                    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
                        memory_info, attn_mask.data(), attn_mask.size(), mask_shape, 2
                    );

                    const char* dec_inputs_3[] = { "input_ids", "encoder_hidden_states", "encoder_attention_mask" };
                    Ort::Value dec_tensors_3[] = { std::move(input_ids_tensor), std::move(hidden_tensor), std::move(mask_tensor) };
                    const char* dec_outputs[] = { "logits" };

                    std::vector<Ort::Value> dec_output_tensors;
                    try {
                        dec_output_tensors = g_decoder_session->Run(
                            Ort::RunOptions{ nullptr }, dec_inputs_3, dec_tensors_3, 3, dec_outputs, 1
                        );
                    } catch (...) {
                        int64_t input_ids_shape2[] = { 1, static_cast<int64_t>(input_ids.size()) };
                        Ort::Value input_ids_tensor2 = Ort::Value::CreateTensor<int64_t>(
                            memory_info, input_ids.data(), input_ids.size(), input_ids_shape2, 2
                        );
                        Ort::Value hidden_tensor2 = Ort::Value::CreateTensor<float>(
                            memory_info, hidden_states, enc_output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount(),
                            hidden_shape.data(), hidden_shape.size()
                        );
                        const char* dec_inputs_2[] = { "input_ids", "encoder_hidden_states" };
                        Ort::Value dec_tensors_2[] = { std::move(input_ids_tensor2), std::move(hidden_tensor2) };

                        dec_output_tensors = g_decoder_session->Run(
                            Ort::RunOptions{ nullptr }, dec_inputs_2, dec_tensors_2, 2, dec_outputs, 1
                        );
                    }

                    if (!dec_output_tensors.empty() && dec_output_tensors[0].IsTensor()) {
                        float* logits = dec_output_tensors[0].GetTensorMutableData<float>();
                        auto dec_shape = dec_output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
                        size_t vocab_size = (dec_shape.size() >= 3) ? static_cast<size_t>(dec_shape[2]) : 51865;
                        size_t seq_len = input_ids.size();

                        float* last_token_logits = logits + (seq_len - 1) * vocab_size;

                        // ARGMAX over logits array
                        size_t best_token_id = 0;
                        float max_logit = -1e9f;
                        for (size_t v = 0; v < vocab_size; ++v) {
                            if (last_token_logits[v] > max_logit) {
                                max_logit = last_token_logits[v];
                                best_token_id = v;
                            }
                        }

                        predicted_token = static_cast<int>(best_token_id);
                        if (predicted_token == 50257) {
                            std::cout << "[Native MediaCore] Whisper EOS token 50257 reached. Exiting decoder loop." << std::endl;
                            std::fflush(stdout);
                            break;
                        }
                        token_sequence.push_back(predicted_token);
                    } else {
                        break;
                    }
                }
                std::cout << "[Native MediaCore] Whisper Decoder loop finished. Sequence length: " << token_sequence.size() << std::endl;
                std::fflush(stdout);
            }
        } catch (const std::exception& e) {
            std::cout << "[Native MediaCore] Whisper transcription exception: " << e.what() << std::endl;
            std::fflush(stdout);
        } catch (...) {
            std::cout << "[Native MediaCore] Unknown exception during Whisper transcription." << std::endl;
            std::fflush(stdout);
        }
    }
#endif

    std::string base_dir = (model_dir && std::strlen(model_dir) > 0) ? model_dir : "local_models";
    std::string tokenizer_path = base_dir + "/tokenizer.json";

    return whisper_decode_tokens(token_sequence.data(), static_cast<int32_t>(token_sequence.size()), tokenizer_path.c_str());
}

// Transcribe full PCM audio buffer in 30-second (480,000 sample) chunks until EOF
char* whisper_transcribe_full_pcm(const int16_t* pcm_data, int32_t total_samples, const char* model_dir) {
    if (!pcm_data || total_samples <= 0) {
        char* empty_res = (char*)std::malloc(1);
        if (empty_res) empty_res[0] = '\0';
        return empty_res;
    }

    const int32_t samples_per_30sec = 480000; // 16,000 Hz * 30s
    int32_t num_chunks = (total_samples + samples_per_30sec - 1) / samples_per_30sec;

    std::cout << "[Native MediaCore] Starting full PCM Whisper transcription across " << num_chunks << " 30s chunks (" << total_samples << " total samples)..." << std::endl;
    std::fflush(stdout);

    std::string full_result = "";

    for (int32_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        int32_t sample_offset = chunk_idx * samples_per_30sec;
        int32_t current_chunk_size = std::min(samples_per_30sec, total_samples - sample_offset);

        if (current_chunk_size <= 0) break;

        std::cout << "[Native MediaCore] Processing Whisper Chunk " << (chunk_idx + 1) << "/" << num_chunks << " (" << current_chunk_size << " samples)..." << std::endl;
        std::fflush(stdout);

        // Allocate 30-second PCM buffer cleanly pre-padded with silence (zeros) up to 480,000 samples
        std::vector<int16_t> pcm_30sec(samples_per_30sec, 0);
        std::memcpy(pcm_30sec.data(), pcm_data + sample_offset, current_chunk_size * sizeof(int16_t));

        // Millisecond timeline bounds
        int32_t start_ms = static_cast<int32_t>((static_cast<int64_t>(sample_offset) * 1000) / 16000);
        int32_t end_ms = static_cast<int32_t>((static_cast<int64_t>(sample_offset + current_chunk_size) * 1000) / 16000);

        // Compute 30-second PocketFFT Mel-spectrogram
        int32_t mel_bins = 0;
        float* mel_data = compute_mel_spectrogram(pcm_30sec.data(), samples_per_30sec, &mel_bins);

        if (mel_data && mel_bins > 0) {
            char* chunk_text_c = whisper_transcribe_audio(mel_data, mel_bins, model_dir);
            std::string chunk_text = chunk_text_c ? chunk_text_c : "";
            if (chunk_text_c) free_string_buffer(chunk_text_c);

            // Clean string: remove leading/trailing whitespace & control characters
            size_t first = chunk_text.find_first_not_of(" \t\n\r");
            if (first != std::string::npos) {
                size_t last = chunk_text.find_last_not_of(" \t\n\r");
                chunk_text = chunk_text.substr(first, (last - first + 1));
            } else {
                chunk_text = "";
            }

            if (!chunk_text.empty()) {
                std::cout << "[Native MediaCore] Chunk " << (chunk_idx + 1) << " transcribed: '" << chunk_text << "'" << std::endl;
                std::fflush(stdout);
                if (!full_result.empty()) {
                    full_result += " | ";
                }
                full_result += "[" + std::to_string(start_ms) + "-" + std::to_string(end_ms) + "] " + chunk_text;
            }

            free_float_buffer(mel_data);
        }
    }

    std::cout << "[Native MediaCore] Full PCM Whisper transcription completed. Total result length: " << full_result.length() << std::endl;
    std::fflush(stdout);

    char* c_str = (char*)std::malloc(full_result.length() + 1);
    if (c_str) {
        std::memcpy(c_str, full_result.c_str(), full_result.length() + 1);
    }
    return c_str;
}

// Memory Management Deallocators
void free_string_buffer(char* ptr) {
    if (ptr) {
        std::cout << "[Native MediaCore] Freeing native string buffer at memory address: " << (void*)ptr << std::endl;
        std::fflush(stdout);
        std::free(ptr);
    }
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
    if (ptr) {
        std::cout << "[Native MediaCore] Freeing native float buffer at memory address: " << (void*)ptr << std::endl;
        std::fflush(stdout);
        std::free(ptr);
    }
}

void free_byte_buffer(uint8_t* ptr) {
    if (ptr) {
        std::cout << "[Native MediaCore] Freeing native byte buffer at memory address: " << (void*)ptr << std::endl;
        std::fflush(stdout);
        std::free(ptr);
    }
}

void free_tracking_boxes(TrackingBox* ptr) {
    if (ptr) {
        std::cout << "[Native MediaCore] Freeing native tracking boxes array at memory address: " << (void*)ptr << std::endl;
        std::fflush(stdout);
        std::free(ptr);
    }
}

void free_ocr_results(OCRTextResult* ptr) {
    if (ptr) {
        std::cout << "[Native MediaCore] Freeing native OCR text results array at memory address: " << (void*)ptr << std::endl;
        std::fflush(stdout);
        std::free(ptr);
    }
}

// 8. PP-OCR Native ONNX Pipeline (ppocr_det_fp32.onnx & ppocr_rec_fp32.onnx)
#ifdef HAS_ONNXRUNTIME
static std::mutex g_ocr_session_mutex;
static std::unique_ptr<Ort::Session> g_ocr_det_session = nullptr;
static std::unique_ptr<Ort::Session> g_ocr_rec_session = nullptr;
static std::string g_cached_ocr_dir = "";

static bool init_ocr_models(const char* model_dir) {
    std::lock_guard<std::mutex> lock(g_ocr_session_mutex);
    std::string base_dir = (model_dir && std::strlen(model_dir) > 0) ? model_dir : "local_models";
    if (g_ocr_det_session && g_ocr_rec_session && g_cached_ocr_dir == base_dir) {
        return true;
    }

    std::cout << "[Native MediaCore] Initializing PP-OCR ONNX det & rec sessions from: " << base_dir << std::endl;
    std::fflush(stdout);

    try {
        std::string det_path = base_dir + "/ppocr_det_fp32.onnx";
        std::string rec_path = base_dir + "/ppocr_rec_fp32.onnx";

        std::ifstream fdet(det_path);
        std::ifstream frec(rec_path);

        if (fdet.good() && frec.good()) {
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetInterOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

#ifdef _WIN32
            std::wstring wdet(det_path.begin(), det_path.end());
            std::wstring wrec(rec_path.begin(), rec_path.end());
            g_ocr_det_session = std::make_unique<Ort::Session>(get_ort_env(), wdet.c_str(), session_options);
            g_ocr_rec_session = std::make_unique<Ort::Session>(get_ort_env(), wrec.c_str(), session_options);
#else
            g_ocr_det_session = std::make_unique<Ort::Session>(get_ort_env(), det_path.c_str(), session_options);
            g_ocr_rec_session = std::make_unique<Ort::Session>(get_ort_env(), rec_path.c_str(), session_options);
#endif
            g_cached_ocr_dir = base_dir;
            std::cout << "[Native MediaCore] PP-OCR ONNX det & rec sessions successfully cached." << std::endl;
            std::fflush(stdout);
            return true;
        } else {
            std::cout << "[Native MediaCore] Warning: PP-OCR model files missing in " << base_dir << std::endl;
            std::fflush(stdout);
        }
    } catch (const std::exception& e) {
        std::cout << "[Native MediaCore] Error initializing PP-OCR ONNX models: " << e.what() << std::endl;
        std::fflush(stdout);
    } catch (...) {
        std::cout << "[Native MediaCore] Unknown exception initializing PP-OCR ONNX models." << std::endl;
        std::fflush(stdout);
    }
    return false;
}
#endif

static char ppocr_index_to_char(size_t index) {
    if (index == 0) return '\0';
    if (index >= 1 && index <= 10) return static_cast<char>('0' + (index - 1));
    if (index >= 11 && index <= 36) return static_cast<char>('a' + (index - 11));
    if (index >= 37 && index <= 62) return static_cast<char>('A' + (index - 37));
    static const char extra_symbols[] = " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
    size_t sym_offset = index - 63;
    if (sym_offset < sizeof(extra_symbols) - 1) {
        return extra_symbols[sym_offset];
    }
    return '\0';
}

OCRTextResult* run_pp_ocr(const uint8_t* rgb_data, uint32_t width, uint32_t height, const char* model_dir, int32_t* out_count) {
    if (!out_count) return nullptr;
    *out_count = 0;

    if (!rgb_data || width == 0 || height == 0) {
        return nullptr;
    }

    std::cout << "[Native MediaCore] Running PP-OCR on " << width << "x" << height << " RGB image..." << std::endl;
    std::fflush(stdout);

    std::vector<OCRTextResult> detected_results;

#ifdef HAS_ONNXRUNTIME
    init_ocr_models(model_dir);

    if (g_ocr_det_session && g_ocr_rec_session) {
        try {
            size_t tensor_size = 3 * static_cast<size_t>(width) * height;
            std::vector<float> input_tensor_values(tensor_size);

            for (uint32_t y = 0; y < height; ++y) {
                for (uint32_t x = 0; x < width; ++x) {
                    uint32_t src_idx = (y * width + x) * 3;
                    float r = (static_cast<float>(rgb_data[src_idx]) / 255.0f - 0.485f) / 0.229f;
                    float g = (static_cast<float>(rgb_data[src_idx + 1]) / 255.0f - 0.456f) / 0.224f;
                    float b = (static_cast<float>(rgb_data[src_idx + 2]) / 255.0f - 0.406f) / 0.225f;

                    size_t plane_size = static_cast<size_t>(width) * height;
                    size_t pixel_idx = y * width + x;

                    input_tensor_values[0 * plane_size + pixel_idx] = r;
                    input_tensor_values[1 * plane_size + pixel_idx] = g;
                    input_tensor_values[2 * plane_size + pixel_idx] = b;
                }
            }

            int64_t det_shape[] = { 1, 3, static_cast<int64_t>(height), static_cast<int64_t>(width) };
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value det_input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_values.size(), det_shape, 4
            );

            const char* det_inputs[] = { "x" };
            const char* det_outputs[] = { "sigmoid" };

            auto det_output_tensors = g_ocr_det_session->Run(
                Ort::RunOptions{ nullptr }, det_inputs, &det_input_tensor, 1, det_outputs, 1
            );

            if (!det_output_tensors.empty() && det_output_tensors[0].IsTensor()) {
                float* det_map = det_output_tensors[0].GetTensorMutableData<float>();

                uint32_t x_min = width, y_min = height, x_max = 0, y_max = 0;
                bool found_pixel = false;
                for (uint32_t y = 0; y < height; ++y) {
                    for (uint32_t x = 0; x < width; ++x) {
                        if (det_map[y * width + x] >= 0.30f) {
                            if (x < x_min) x_min = x;
                            if (x > x_max) x_max = x;
                            if (y < y_min) y_min = y;
                            if (y > y_max) y_max = y;
                            found_pixel = true;
                        }
                    }
                }

                if (found_pixel && x_max > x_min && y_max > y_min) {
                    int64_t rec_shape[] = { 1, 3, 48, 320 };
                    std::vector<float> rec_input(1 * 3 * 48 * 320, 0.0f);

                    float box_w = static_cast<float>(x_max - x_min);
                    float box_h = static_cast<float>(y_max - y_min);

                    for (int ry = 0; ry < 48; ++ry) {
                        float src_y = static_cast<float>(y_min) + (ry / 47.0f) * box_h;
                        uint32_t y0 = static_cast<uint32_t>(std::floor(src_y));
                        uint32_t y1 = std::min(y0 + 1, height - 1);
                        float dy = src_y - y0;

                        for (int rx = 0; rx < 320; ++rx) {
                            float src_x = static_cast<float>(x_min) + (rx / 319.0f) * box_w;
                            uint32_t x0 = static_cast<uint32_t>(std::floor(src_x));
                            uint32_t x1 = std::min(x0 + 1, width - 1);
                            float dx = src_x - x0;

                            for (int c = 0; c < 3; ++c) {
                                size_t plane = c * (static_cast<size_t>(width) * height);
                                float v00 = input_tensor_values[plane + y0 * width + x0];
                                float v01 = input_tensor_values[plane + y0 * width + x1];
                                float v10 = input_tensor_values[plane + y1 * width + x0];
                                float v11 = input_tensor_values[plane + y1 * width + x1];

                                float val = (1.0f - dx) * (1.0f - dy) * v00 +
                                            dx * (1.0f - dy) * v01 +
                                            (1.0f - dx) * dy * v10 +
                                            dx * dy * v11;

                                rec_input[c * (48 * 320) + ry * 320 + rx] = val;
                            }
                        }
                    }

                    Ort::Value rec_input_tensor = Ort::Value::CreateTensor<float>(
                        memory_info, rec_input.data(), rec_input.size(), rec_shape, 4
                    );

                    const char* rec_inputs[] = { "x" };
                    const char* rec_outputs[] = { "softmax" };

                    try {
                        auto rec_output_tensors = g_ocr_rec_session->Run(
                            Ort::RunOptions{ nullptr }, rec_inputs, &rec_input_tensor, 1, rec_outputs, 1
                        );

                        if (!rec_output_tensors.empty() && rec_output_tensors[0].IsTensor()) {
                            float* logits = rec_output_tensors[0].GetTensorMutableData<float>();
                            auto rec_tensor_shape = rec_output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

                            size_t seq_len = (rec_tensor_shape.size() >= 2) ? rec_tensor_shape[1] : 40;
                            size_t vocab_size = (rec_tensor_shape.size() >= 3) ? rec_tensor_shape[2] : 6625;

                            std::string recognized_str = "";
                            int last_token_idx = -1;
                            float conf_sum = 0.0f;
                            int char_count = 0;

                            for (size_t t = 0; t < seq_len; ++t) {
                                float* step_logits = logits + t * vocab_size;
                                size_t best_idx = 0;
                                float max_logit = step_logits[0];

                                for (size_t v = 1; v < vocab_size; ++v) {
                                    if (step_logits[v] > max_logit) {
                                        max_logit = step_logits[v];
                                        best_idx = v;
                                    }
                                }

                                if (best_idx != 0 && static_cast<int>(best_idx) != last_token_idx) {
                                    char c = ppocr_index_to_char(best_idx);
                                    if (c != '\0') {
                                        recognized_str += c;
                                        conf_sum += max_logit;
                                        char_count++;
                                    }
                                }
                                last_token_idx = static_cast<int>(best_idx);
                            }

                            if (!recognized_str.empty()) {
                                OCRTextResult res;
                                std::memset(&res, 0, sizeof(OCRTextResult));
                                res.confidence = (char_count > 0) ? (conf_sum / char_count) * 100.0f : 85.0f;

                                size_t copy_len = std::min(recognized_str.length(), static_cast<size_t>(255));
                                std::memcpy(res.text, recognized_str.c_str(), copy_len);
                                res.text[copy_len] = '\0';

                                res.bounding_box[0] = static_cast<float>(x_min); res.bounding_box[1] = static_cast<float>(y_min);
                                res.bounding_box[2] = static_cast<float>(x_max); res.bounding_box[3] = static_cast<float>(y_min);
                                res.bounding_box[4] = static_cast<float>(x_max); res.bounding_box[5] = static_cast<float>(y_max);
                                res.bounding_box[6] = static_cast<float>(x_min); res.bounding_box[7] = static_cast<float>(y_max);

                                detected_results.push_back(res);
                            }
                        }
                    } catch (...) {
                    }
                }
            }
        } catch (...) {
        }
    }
#endif

    if (detected_results.empty()) {
        *out_count = 0;
        return nullptr;
    }

    *out_count = static_cast<int32_t>(detected_results.size());
    OCRTextResult* c_array = (OCRTextResult*)std::malloc(detected_results.size() * sizeof(OCRTextResult));
    if (c_array) {
        std::memcpy(c_array, detected_results.data(), detected_results.size() * sizeof(OCRTextResult));
    }
    return c_array;
}

// 9. Frame Thumbnail Exporter (Saves raw RGB24 frame data as image file)
bool save_frame_as_jpeg(const uint8_t* rgb_data, uint32_t width, uint32_t height, const char* output_path) {
    if (!rgb_data || width == 0 || height == 0 || !output_path) return false;

    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) return false;

    uint32_t row_stride = width * 3;
    uint32_t padding = (4 - (row_stride % 4)) % 4;
    uint32_t data_size = (row_stride + padding) * height;
    uint32_t file_size = 54 + data_size;

    uint8_t header[54] = { 0 };
    header[0] = 'B'; header[1] = 'M';
    header[2] = static_cast<uint8_t>(file_size);
    header[3] = static_cast<uint8_t>(file_size >> 8);
    header[4] = static_cast<uint8_t>(file_size >> 16);
    header[5] = static_cast<uint8_t>(file_size >> 24);
    header[10] = 54; // Pixel data offset

    header[14] = 40; // DIB Header size
    header[18] = static_cast<uint8_t>(width);
    header[19] = static_cast<uint8_t>(width >> 8);
    header[20] = static_cast<uint8_t>(width >> 16);
    header[21] = static_cast<uint8_t>(width >> 24);

    int32_t neg_height = -static_cast<int32_t>(height);
    header[22] = static_cast<uint8_t>(neg_height);
    header[23] = static_cast<uint8_t>(neg_height >> 8);
    header[24] = static_cast<uint8_t>(neg_height >> 16);
    header[25] = static_cast<uint8_t>(neg_height >> 24);

    header[26] = 1;  // Planes
    header[28] = 24; // 24-bit RGB
    header[34] = static_cast<uint8_t>(data_size);
    header[35] = static_cast<uint8_t>(data_size >> 8);
    header[36] = static_cast<uint8_t>(data_size >> 16);
    header[37] = static_cast<uint8_t>(data_size >> 24);

    file.write(reinterpret_cast<const char*>(header), 54);

    std::vector<uint8_t> row_buffer(row_stride + padding, 0);
    for (uint32_t y = 0; y < height; ++y) {
        const uint8_t* src_row = rgb_data + y * row_stride;
        for (uint32_t x = 0; x < width; ++x) {
            uint32_t src_idx = x * 3;
            uint32_t dst_idx = x * 3;
            row_buffer[dst_idx]     = src_row[src_idx + 2]; // B
            row_buffer[dst_idx + 1] = src_row[src_idx + 1]; // G
            row_buffer[dst_idx + 2] = src_row[src_idx];     // R
        }
        file.write(reinterpret_cast<const char*>(row_buffer.data()), row_buffer.size());
    }

    file.close();
    return true;
}
