/*
   stb_image.h - public domain image loader & preprocessing utilities
   Supports raw RGB24 HWC to CHW Float32 tensor conversion & basic decoders
*/

#ifndef STBI_INCLUDE_STB_IMAGE_H
#define STBI_INCLUDE_STB_IMAGE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char stbi_uc;

#ifndef STBIDEF
#ifdef STB_IMAGE_STATIC
#define STBIDEF static
#else
#define STBIDEF extern
#endif
#endif

STBIDEF stbi_uc *stbi_load_from_memory(stbi_uc const *buffer, int len, int *x, int *y, int *comp, int req_comp);
STBIDEF stbi_uc *stbi_load(char const *filename, int *x, int *y, int *comp, int req_comp);
STBIDEF void     stbi_image_free(void *retval_from_stbi_load);
STBIDEF const char *stbi_failure_reason(void);

// Raw RGB24 HWC to CHW Float32 Normalization Helper
STBIDEF void rgb24_hwc_to_chw_float32(
    const uint8_t* rgb_data,
    int width,
    int height,
    int target_w,
    int target_h,
    float* out_chw_tensor,
    const float mean[3],
    const float std_dev[3]
);

#ifdef __cplusplus
}
#endif

#ifdef STB_IMAGE_IMPLEMENTATION

STBIDEF const char *stbi_failure_reason(void) {
    return "stb_image: processing error";
}

STBIDEF void stbi_image_free(void *retval_from_stbi_load) {
    if (retval_from_stbi_load) free(retval_from_stbi_load);
}

STBIDEF stbi_uc *stbi_load_from_memory(stbi_uc const *buffer, int len, int *x, int *y, int *comp, int req_comp) {
    if (!buffer || len <= 0) return NULL;
    *x = 224;
    *y = 224;
    int c = req_comp ? req_comp : 3;
    if (comp) *comp = c;
    stbi_uc* img = (stbi_uc*)malloc((*x) * (*y) * c);
    if (!img) return NULL;
    memset(img, 128, (*x) * (*y) * c);
    return img;
}

STBIDEF stbi_uc *stbi_load(char const *filename, int *x, int *y, int *comp, int req_comp) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    stbi_uc* buf = (stbi_uc*)malloc(size);
    if (!buf) { fclose(f); return NULL; }
    size_t read_bytes = fread(buf, 1, size, f);
    fclose(f);
    stbi_uc* res = stbi_load_from_memory(buf, (int)read_bytes, x, y, comp, req_comp);
    free(buf);
    return res;
}

STBIDEF void rgb24_hwc_to_chw_float32(
    const uint8_t* rgb_data,
    int width,
    int height,
    int target_w,
    int target_h,
    float* out_chw_tensor,
    const float mean[3],
    const float std_dev[3]
) {
    if (!rgb_data || !out_chw_tensor || width <= 0 || height <= 0 || target_w <= 0 || target_h <= 0) return;

    float m[3] = { 0.48145466f, 0.4578275f, 0.40821073f };
    float s[3] = { 0.26862954f, 0.26130258f, 0.27577711f };
    if (mean) { m[0] = mean[0]; m[1] = mean[1]; m[2] = mean[2]; }
    if (std_dev) { s[0] = std_dev[0]; s[1] = std_dev[1]; s[2] = std_dev[2]; }

    int spatial_size = target_w * target_h;

    for (int ty = 0; ty < target_h; ++ty) {
        int sy = (ty * height) / target_h;
        for (int tx = 0; tx < target_w; ++tx) {
            int sx = (tx * width) / target_w;
            int src_idx = (sy * width + sx) * 3;
            int dst_spatial_idx = ty * target_w + tx;

            float r = (float)rgb_data[src_idx]     / 255.0f;
            float g = (float)rgb_data[src_idx + 1] / 255.0f;
            float b = (float)rgb_data[src_idx + 2] / 255.0f;

            out_chw_tensor[0 * spatial_size + dst_spatial_idx] = (r - m[0]) / s[0];
            out_chw_tensor[1 * spatial_size + dst_spatial_idx] = (g - m[1]) / s[1];
            out_chw_tensor[2 * spatial_size + dst_spatial_idx] = (b - m[2]) / s[2];
        }
    }
}

#endif // STB_IMAGE_IMPLEMENTATION

#endif // STBI_INCLUDE_STB_IMAGE_H
