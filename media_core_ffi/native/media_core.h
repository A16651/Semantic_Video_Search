#ifndef MEDIA_CORE_H
#define MEDIA_CORE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// Frame structure representing decoded video frame
struct VideoFrame {
    uint32_t width;
    uint32_t height;
    uint32_t timestamp_ms;
    uint8_t* rgb_data; // rgb24 raw format
};

// Object detection / Face tracking bounding box
struct TrackingBox {
    int32_t id;
    char label[64];
    float confidence;
    float x_min;
    float y_min;
    float x_max;
    float y_max;
};

// Extracted text structure from PP-OCR
struct OCRTextResult {
    char text[256];
    float confidence;
    float bounding_box[8]; // 4 points x,y coordinates
};

// 1. Scene Change / Frame Extraction (SAD Pixel-Diffing)
EXPORT bool compute_sad_threshold(const uint8_t* frame_a, const uint8_t* frame_b, uint32_t width, uint32_t height, float threshold_percentage);

// 2. ONNX SigLIP Text-Encoding Pipeline (Enforced to exactly 512 dimensions)
EXPORT float* encode_text_query(const char* query, int32_t* out_dimension);

// 3. ONNX SigLIP Image Projection (Asserts & Down-projects high-dim to 512 dimensions)
EXPORT float* project_image_embedding(const float* raw_1152_embedding, int32_t* out_dimension);

// 4. Whisper Tiny PocketFFT Audio Engine & ONNX Autoregressive Decoder
EXPORT bool init_whisper_models(const char* model_dir);
EXPORT float* compute_mel_spectrogram(const int16_t* pcm_data, int32_t sample_count, int32_t* out_bin_count);
EXPORT float* whisper_compute_mel(const int16_t* pcm_data, int32_t sample_count, int32_t* out_bin_count);
EXPORT char* whisper_transcribe_audio(const float* mel_data, int32_t mel_bins, const char* model_dir);
EXPORT char* whisper_transcribe_full_pcm(const int16_t* pcm_data, int32_t total_samples, const char* model_dir);

// 5. Whisper Token Decoder (nlohmann::json tokenizer.json parser mapping token IDs to English string)
EXPORT char* whisper_decode_tokens(const int32_t* token_ids, int32_t token_count, const char* tokenizer_json_path);

// 6. Image Frame Normalization (RGB24 HWC to CHW Float32)
EXPORT float* normalize_rgb24_hwc_to_chw(const uint8_t* rgb_data, uint32_t width, uint32_t height, uint32_t target_w, uint32_t target_h);

// 7. Memory Management (Explicit Native Deallocators)
EXPORT void free_float_buffer(float* ptr);
EXPORT void free_byte_buffer(uint8_t* ptr);
EXPORT void free_string_buffer(char* ptr);
EXPORT void free_tracking_boxes(TrackingBox* ptr);
EXPORT void free_ocr_results(OCRTextResult* ptr);

// 8. PP-OCR Native ONNX Pipeline
EXPORT OCRTextResult* run_pp_ocr(const uint8_t* rgb_data, uint32_t width, uint32_t height, const char* model_dir, int32_t* out_count);

// 9. Frame Thumbnail Exporter
EXPORT bool save_frame_as_jpeg(const uint8_t* rgb_data, uint32_t width, uint32_t height, const char* output_path);

// 10. SigLIP Vision ONNX Transformer Encoder (CHW Float32 [1, 3, 224, 224] -> 512-dim visual vector)
EXPORT float* encode_image_frame(const float* chw_data, int32_t* out_dimension);

}

#endif // MEDIA_CORE_H
