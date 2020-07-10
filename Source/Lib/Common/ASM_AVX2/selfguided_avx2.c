/*
 * Copyright (c) 2018, Alliance for Open Media. All rights reserved
 *
 * This source code is subject to the terms of the BSD 2 Clause License and
 * the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
 * was not distributed with this source code in the LICENSE file, you can
 * obtain it at www.aomedia.org/license/software. If the Alliance for Open
 * Media Patent License 1.0 was not distributed with this source code in the
 * PATENTS file, you can obtain it at www.aomedia.org/license/patent.
 */

#include "EbDefinitions.h"
#include <immintrin.h>
#include "common_dsp_rtcd.h"
#include "EbRestoration.h"
#include "synonyms.h"
#include "synonyms_avx2.h"
#include "transpose_avx2.h"
#include "transpose_sse2.h"
#include "avx512fintrin.h"


static INLINE void cvt_16to32bit_8x8(const __m128i s[8], __m256i r[8]) {
    r[0] = _mm256_cvtepu16_epi32(s[0]);
    r[1] = _mm256_cvtepu16_epi32(s[1]);
    r[2] = _mm256_cvtepu16_epi32(s[2]);
    r[3] = _mm256_cvtepu16_epi32(s[3]);
    r[4] = _mm256_cvtepu16_epi32(s[4]);
    r[5] = _mm256_cvtepu16_epi32(s[5]);
    r[6] = _mm256_cvtepu16_epi32(s[6]);
    r[7] = _mm256_cvtepu16_epi32(s[7]);
}

static INLINE void add_32bit_8x8(const __m256i neighbor, __m256i r[8]) {
    r[0] = _mm256_add_epi32(neighbor, r[0]);
    r[1] = _mm256_add_epi32(r[0], r[1]);
    r[2] = _mm256_add_epi32(r[1], r[2]);
    r[3] = _mm256_add_epi32(r[2], r[3]);
    r[4] = _mm256_add_epi32(r[3], r[4]);
    r[5] = _mm256_add_epi32(r[4], r[5]);
    r[6] = _mm256_add_epi32(r[5], r[6]);
    r[7] = _mm256_add_epi32(r[6], r[7]);
}

static INLINE void store_32bit_8x8(const __m256i r[8], int32_t *const buf,
                                   const int32_t buf_stride) {
    _mm256_storeu_si256((__m256i *)(buf + 0 * buf_stride), r[0]);
    _mm256_storeu_si256((__m256i *)(buf + 1 * buf_stride), r[1]);
    _mm256_storeu_si256((__m256i *)(buf + 2 * buf_stride), r[2]);
    _mm256_storeu_si256((__m256i *)(buf + 3 * buf_stride), r[3]);
    _mm256_storeu_si256((__m256i *)(buf + 4 * buf_stride), r[4]);
    _mm256_storeu_si256((__m256i *)(buf + 5 * buf_stride), r[5]);
    _mm256_storeu_si256((__m256i *)(buf + 6 * buf_stride), r[6]);
    _mm256_storeu_si256((__m256i *)(buf + 7 * buf_stride), r[7]);
}

static AOM_FORCE_INLINE void integral_images(const uint8_t *src, int32_t src_stride, int32_t width,
                                             int32_t height, int32_t *C, int32_t *D,
                                             int32_t buf_stride) {
    const uint8_t *src_t = src;
    int32_t *      ct    = C + buf_stride + 1;
    int32_t *      dt    = D + buf_stride + 1;
    const __m256i  zero  = _mm256_setzero_si256();

    memset(C, 0, sizeof(*C) * (width + 8));
    memset(D, 0, sizeof(*D) * (width + 8));

    int y = 0;
    do {
        __m256i c_left = _mm256_setzero_si256();
        __m256i d_left = _mm256_setzero_si256();

        // zero the left column.
        ct[0 * buf_stride - 1] = dt[0 * buf_stride - 1] = 0;
        ct[1 * buf_stride - 1] = dt[1 * buf_stride - 1] = 0;
        ct[2 * buf_stride - 1] = dt[2 * buf_stride - 1] = 0;
        ct[3 * buf_stride - 1] = dt[3 * buf_stride - 1] = 0;
        ct[4 * buf_stride - 1] = dt[4 * buf_stride - 1] = 0;
        ct[5 * buf_stride - 1] = dt[5 * buf_stride - 1] = 0;
        ct[6 * buf_stride - 1] = dt[6 * buf_stride - 1] = 0;
        ct[7 * buf_stride - 1] = dt[7 * buf_stride - 1] = 0;

        int x = 0;
        do {
            __m128i s[8];
            __m256i r32[8];

            s[0] = _mm_loadl_epi64((__m128i *)(src_t + 0 * src_stride + x));
            s[1] = _mm_loadl_epi64((__m128i *)(src_t + 1 * src_stride + x));
            s[2] = _mm_loadl_epi64((__m128i *)(src_t + 2 * src_stride + x));
            s[3] = _mm_loadl_epi64((__m128i *)(src_t + 3 * src_stride + x));
            s[4] = _mm_loadl_epi64((__m128i *)(src_t + 4 * src_stride + x));
            s[5] = _mm_loadl_epi64((__m128i *)(src_t + 5 * src_stride + x));
            s[6] = _mm_loadl_epi64((__m128i *)(src_t + 6 * src_stride + x));
            s[7] = _mm_loadl_epi64((__m128i *)(src_t + 7 * src_stride + x));

            partial_transpose_8bit_8x8(s, s);

            s[7] = _mm_unpackhi_epi8(s[3], _mm_setzero_si128());
            s[6] = _mm_unpacklo_epi8(s[3], _mm_setzero_si128());
            s[5] = _mm_unpackhi_epi8(s[2], _mm_setzero_si128());
            s[4] = _mm_unpacklo_epi8(s[2], _mm_setzero_si128());
            s[3] = _mm_unpackhi_epi8(s[1], _mm_setzero_si128());
            s[2] = _mm_unpacklo_epi8(s[1], _mm_setzero_si128());
            s[1] = _mm_unpackhi_epi8(s[0], _mm_setzero_si128());
            s[0] = _mm_unpacklo_epi8(s[0], _mm_setzero_si128());

            cvt_16to32bit_8x8(s, r32);
            add_32bit_8x8(d_left, r32);
            d_left = r32[7];

            transpose_32bit_8x8_avx2(r32, r32);

            const __m256i d_top = _mm256_loadu_si256((__m256i *)(dt - buf_stride + x));
            add_32bit_8x8(d_top, r32);
            store_32bit_8x8(r32, dt + x, buf_stride);

            s[0] = _mm_mullo_epi16(s[0], s[0]);
            s[1] = _mm_mullo_epi16(s[1], s[1]);
            s[2] = _mm_mullo_epi16(s[2], s[2]);
            s[3] = _mm_mullo_epi16(s[3], s[3]);
            s[4] = _mm_mullo_epi16(s[4], s[4]);
            s[5] = _mm_mullo_epi16(s[5], s[5]);
            s[6] = _mm_mullo_epi16(s[6], s[6]);
            s[7] = _mm_mullo_epi16(s[7], s[7]);

            cvt_16to32bit_8x8(s, r32);
            add_32bit_8x8(c_left, r32);
            c_left = r32[7];

            transpose_32bit_8x8_avx2(r32, r32);

            const __m256i c_top = _mm256_loadu_si256((__m256i *)(ct - buf_stride + x));
            add_32bit_8x8(c_top, r32);
            store_32bit_8x8(r32, ct + x, buf_stride);
            x += 8;
        } while (x < width);

        /* Used in calc_ab and calc_ab_fast, when calc out of right border */
        for (int ln = 0; ln < 8; ++ln) {
            _mm256_storeu_si256((__m256i *)(ct + x + ln * buf_stride), zero);
            _mm256_storeu_si256((__m256i *)(dt + x + ln * buf_stride), zero);
        }

        src_t += 8 * src_stride;
        ct += 8 * buf_stride;
        dt += 8 * buf_stride;
        y += 8;
    } while (y < height);
}

static AOM_FORCE_INLINE void integral_images_highbd(const uint16_t *src, int32_t src_stride,
                                                    int32_t width, int32_t height, int32_t *C,
                                                    int32_t *D, int32_t buf_stride) {
                                                        
    const uint16_t *src_t = src;
    int32_t *       ct    = C + buf_stride + 1;
    int32_t *       dt    = D + buf_stride + 1;
    const __m256i   zero  = _mm256_setzero_si256();

    memset(C, 0, sizeof(*C) * (width + 8));
    memset(D, 0, sizeof(*D) * (width + 8));

    int y = 0;
    do {
        __m256i c_left = _mm256_setzero_si256();
        __m256i d_left = _mm256_setzero_si256();

        // zero the left column.
        ct[0 * buf_stride - 1] = dt[0 * buf_stride - 1] = 0;
        ct[1 * buf_stride - 1] = dt[1 * buf_stride - 1] = 0;
        ct[2 * buf_stride - 1] = dt[2 * buf_stride - 1] = 0;
        ct[3 * buf_stride - 1] = dt[3 * buf_stride - 1] = 0;
        ct[4 * buf_stride - 1] = dt[4 * buf_stride - 1] = 0;
        ct[5 * buf_stride - 1] = dt[5 * buf_stride - 1] = 0;
        ct[6 * buf_stride - 1] = dt[6 * buf_stride - 1] = 0;
        ct[7 * buf_stride - 1] = dt[7 * buf_stride - 1] = 0;

        int x = 0;
        do {
            __m128i s[8];
            __m256i r32[8], a32[8];

            s[0] = _mm_loadu_si128((__m128i *)(src_t + 0 * src_stride + x));
            s[1] = _mm_loadu_si128((__m128i *)(src_t + 1 * src_stride + x));
            s[2] = _mm_loadu_si128((__m128i *)(src_t + 2 * src_stride + x));
            s[3] = _mm_loadu_si128((__m128i *)(src_t + 3 * src_stride + x));
            s[4] = _mm_loadu_si128((__m128i *)(src_t + 4 * src_stride + x));
            s[5] = _mm_loadu_si128((__m128i *)(src_t + 5 * src_stride + x));
            s[6] = _mm_loadu_si128((__m128i *)(src_t + 6 * src_stride + x));
            s[7] = _mm_loadu_si128((__m128i *)(src_t + 7 * src_stride + x));

            transpose_16bit_8x8(s, s);

            cvt_16to32bit_8x8(s, r32);

            a32[0] = _mm256_madd_epi16(r32[0], r32[0]);
            a32[1] = _mm256_madd_epi16(r32[1], r32[1]);
            a32[2] = _mm256_madd_epi16(r32[2], r32[2]);
            a32[3] = _mm256_madd_epi16(r32[3], r32[3]);
            a32[4] = _mm256_madd_epi16(r32[4], r32[4]);
            a32[5] = _mm256_madd_epi16(r32[5], r32[5]);
            a32[6] = _mm256_madd_epi16(r32[6], r32[6]);
            a32[7] = _mm256_madd_epi16(r32[7], r32[7]);

            add_32bit_8x8(c_left, a32);
            c_left = a32[7];

            transpose_32bit_8x8_avx2(a32, a32);

            const __m256i c_top = _mm256_loadu_si256((__m256i *)(ct - buf_stride + x));
            add_32bit_8x8(c_top, a32);
            store_32bit_8x8(a32, ct + x, buf_stride);

            add_32bit_8x8(d_left, r32);
            d_left = r32[7];

            transpose_32bit_8x8_avx2(r32, r32);

            const __m256i d_top = _mm256_loadu_si256((__m256i *)(dt - buf_stride + x));
            add_32bit_8x8(d_top, r32);
            store_32bit_8x8(r32, dt + x, buf_stride);
            x += 8;
        } while (x < width);

        /* Used in calc_ab and calc_ab_fast, when calc out of right border */
        for (int ln = 0; ln < 8; ++ln) {
            _mm256_storeu_si256((__m256i *)(ct + x + ln * buf_stride), zero);
            _mm256_storeu_si256((__m256i *)(dt + x + ln * buf_stride), zero);
        }

        src_t += 8 * src_stride;
        ct += 8 * buf_stride;
        dt += 8 * buf_stride;
        y += 8;
    } while (y < height);
}

static INLINE __m512i _mm512_extract4x128(__m512i a, __m512i b, __m512i c, __m512i d, const uint8_t n){
    if(n == 0){
        __m256i s = _mm256_inserti64x2(_mm256_castsi128_si256(_mm512_extracti64x2_epi64(a, 0b0)), _mm512_extracti64x2_epi64(b, 0b0), 0b1);
        __m256i t = _mm256_inserti64x2(_mm256_castsi128_si256(_mm512_extracti64x2_epi64(c, 0b0)), _mm512_extracti64x2_epi64(d, 0b0), 0b1);
        return _mm512_inserti64x4(_mm512_castsi256_si512(s), t, 1);
    } else if(n ==1){
        __m256i s = _mm256_inserti64x2(_mm256_castsi128_si256(_mm512_extracti64x2_epi64(a, 0b1)), _mm512_extracti64x2_epi64(b, 0b1), 0b1);
        __m256i t = _mm256_inserti64x2(_mm256_castsi128_si256(_mm512_extracti64x2_epi64(c, 0b1)), _mm512_extracti64x2_epi64(d, 0b1), 0b1);
        return _mm512_inserti64x4(_mm512_castsi256_si512(s), t, 1);
    } else if(n == 2){
        __m256i s = _mm256_inserti64x2(_mm256_castsi128_si256(_mm512_extracti64x2_epi64(a, 0b10)), _mm512_extracti64x2_epi64(b, 0b10), 0b1);
        __m256i t = _mm256_inserti64x2(_mm256_castsi128_si256(_mm512_extracti64x2_epi64(c, 0b10)), _mm512_extracti64x2_epi64(d, 0b10), 0b1);
        return _mm512_inserti64x4(_mm512_castsi256_si512(s), t, 1);
    } else if(n == 3){
        __m256i s = _mm256_inserti64x2(_mm256_castsi128_si256(_mm512_extracti64x2_epi64(a, 0b11)), _mm512_extracti64x2_epi64(b, 0b11), 0b1);
        __m256i t = _mm256_inserti64x2(_mm256_castsi128_si256(_mm512_extracti64x2_epi64(c, 0b11)), _mm512_extracti64x2_epi64(d, 0b11), 0b1);
        return _mm512_inserti64x4(_mm512_castsi256_si512(s), t, 1);
    }else {
        return _mm512_setzero_si512();
    }
}

static INLINE void transpose_32bit_16x16_avx512(const __m512i *const in, __m512i *const out) {
    __m512i b[16];

    for (int i = 0; i < 16; i += 2) {
        b[i / 2 + 0] = _mm512_unpacklo_epi32(in[i], in[i + 1]);
        b[i / 2 + 8] = _mm512_unpackhi_epi32(in[i], in[i + 1]);
    }
    __m512i c[16];
    for (int i = 0; i < 16; i += 2) {
        c[i / 2 + 0] = _mm512_unpacklo_epi64(b[i], b[i + 1]);
        c[i / 2 + 8] = _mm512_unpackhi_epi64(b[i], b[i + 1]);
    }
    
    for(int i = 0; i < 16; i+=4){
        out[i] = _mm512_extract4x128(c[0], c[1], c[2], c[3], i/4);
        out[i] = _mm512_extract4x128(c[8], c[9], c[10], c[11], i/4);
        out[i] = _mm512_extract4x128(c[4], c[5], c[6], c[7], i/4);
        out[i] = _mm512_extract4x128(c[12], c[13], c[14], c[15], i/4);
    }

        out[0] = _mm512_extract4x128(c[0], c[1], c[2], c[3], 0);
        out[1] = _mm512_extract4x128(c[8], c[9], c[10], c[11], 0);
        out[2] = _mm512_extract4x128(c[4], c[5], c[6], c[7], 0);
        out[3] = _mm512_extract4x128(c[12], c[13], c[14], c[15], 0);
        out[4] = _mm512_extract4x128(c[0], c[1], c[2], c[3], 1);
        out[5] = _mm512_extract4x128(c[8], c[9], c[10], c[11], 1);
        out[6] = _mm512_extract4x128(c[4], c[5], c[6], c[7], 1);
        out[7] = _mm512_extract4x128(c[12], c[13], c[14], c[15], 1);
        out[8] = _mm512_extract4x128(c[0], c[1], c[2], c[3], 2);
        out[9] = _mm512_extract4x128(c[8], c[9], c[10], c[11], 2);
        out[10] = _mm512_extract4x128(c[4], c[5], c[6], c[7], 2);
        out[11] = _mm512_extract4x128(c[12], c[13], c[14], c[15], 2);
        out[12] = _mm512_extract4x128(c[0], c[1], c[2], c[3], 3);
        out[13] = _mm512_extract4x128(c[8], c[9], c[10], c[11], 3);
        out[14] = _mm512_extract4x128(c[4], c[5], c[6], c[7], 3);
        out[15] = _mm512_extract4x128(c[12], c[13], c[14], c[15], 3);
}

static INLINE void add_32bit_16x16(const __m512i neighbor, __m512i r[16]) {
    r[0] = _mm512_add_epi32(neighbor, r[0]);
    r[1] = _mm512_add_epi32(r[0], r[1]);
    r[2] = _mm512_add_epi32(r[1], r[2]);
    r[3] = _mm512_add_epi32(r[2], r[3]);
    r[4] = _mm512_add_epi32(r[3], r[4]);
    r[5] = _mm512_add_epi32(r[4], r[5]);
    r[6] = _mm512_add_epi32(r[5], r[6]);
    r[7] = _mm512_add_epi32(r[6], r[7]);
    r[8] = _mm512_add_epi32(r[7], r[8]);
    r[9] = _mm512_add_epi32(r[8], r[9]);
    r[10] = _mm512_add_epi32(r[9], r[10]);
    r[11] = _mm512_add_epi32(r[10], r[11]);
    r[12] = _mm512_add_epi32(r[11], r[12]);
    r[13] = _mm512_add_epi32(r[12], r[13]);
    r[14] = _mm512_add_epi32(r[13], r[14]);
    r[15] = _mm512_add_epi32(r[14], r[15]);
}

static INLINE void store_32bit_16x16(const __m512i r[16], int32_t *const buf,
                                   const int32_t buf_stride) {
    _mm512_storeu_si512((__m512i *)(buf + 0 * buf_stride), r[0]);
    _mm512_storeu_si512((__m512i *)(buf + 1 * buf_stride), r[1]);
    _mm512_storeu_si512((__m512i *)(buf + 2 * buf_stride), r[2]);
    _mm512_storeu_si512((__m512i *)(buf + 3 * buf_stride), r[3]);
    _mm512_storeu_si512((__m512i *)(buf + 4 * buf_stride), r[4]);
    _mm512_storeu_si512((__m512i *)(buf + 5 * buf_stride), r[5]);
    _mm512_storeu_si512((__m512i *)(buf + 6 * buf_stride), r[6]);
    _mm512_storeu_si512((__m512i *)(buf + 7 * buf_stride), r[7]);
    _mm512_storeu_si512((__m512i *)(buf + 8 * buf_stride), r[8]);
    _mm512_storeu_si512((__m512i *)(buf + 9 * buf_stride), r[9]);
    _mm512_storeu_si512((__m512i *)(buf + 10 * buf_stride), r[10]);
    _mm512_storeu_si512((__m512i *)(buf + 11 * buf_stride), r[11]);
    _mm512_storeu_si512((__m512i *)(buf + 12 * buf_stride), r[12]);
    _mm512_storeu_si512((__m512i *)(buf + 13 * buf_stride), r[13]);
    _mm512_storeu_si512((__m512i *)(buf + 14 * buf_stride), r[14]);
    _mm512_storeu_si512((__m512i *)(buf + 15 * buf_stride), r[15]);
}

static INLINE void cvt_16to32bit_16x16(const __m256i s[16], __m512i r[16]) {
    r[0] = _mm512_cvtepu16_epi32(s[0]);
    r[1] = _mm512_cvtepu16_epi32(s[1]);
    r[2] = _mm512_cvtepu16_epi32(s[2]);
    r[3] = _mm512_cvtepu16_epi32(s[3]);
    r[4] = _mm512_cvtepu16_epi32(s[4]);
    r[5] = _mm512_cvtepu16_epi32(s[5]);
    r[6] = _mm512_cvtepu16_epi32(s[6]);
    r[7] = _mm512_cvtepu16_epi32(s[7]);
    r[8] = _mm512_cvtepu16_epi32(s[8]);
    r[9] = _mm512_cvtepu16_epi32(s[9]);
    r[10] = _mm512_cvtepu16_epi32(s[10]);
    r[11] = _mm512_cvtepu16_epi32(s[11]);
    r[12] = _mm512_cvtepu16_epi32(s[12]);
    r[13] = _mm512_cvtepu16_epi32(s[13]);
    r[14] = _mm512_cvtepu16_epi32(s[14]);
    r[15] = _mm512_cvtepu16_epi32(s[15]);
}

static __inline void transpose_16bit_16x16_avx2(const __m256i *const in, __m256i *const out) {
    // Unpack 16 bit elements. Goes from:
    // in[0]: 00 01 02 03  08 09 0a 0b  04 05 06 07  0c 0d 0e 0f
    // in[1]: 10 11 12 13  18 19 1a 1b  14 15 16 17  1c 1d 1e 1f
    // in[2]: 20 21 22 23  28 29 2a 2b  24 25 26 27  2c 2d 2e 2f
    // in[3]: 30 31 32 33  38 39 3a 3b  34 35 36 37  3c 3d 3e 3f
    // in[4]: 40 41 42 43  48 49 4a 4b  44 45 46 47  4c 4d 4e 4f
    // in[5]: 50 51 52 53  58 59 5a 5b  54 55 56 57  5c 5d 5e 5f
    // in[6]: 60 61 62 63  68 69 6a 6b  64 65 66 67  6c 6d 6e 6f
    // in[7]: 70 71 72 73  78 79 7a 7b  74 75 76 77  7c 7d 7e 7f
    // in[8]: 80 81 82 83  88 89 8a 8b  84 85 86 87  8c 8d 8e 8f
    // to:
    // a0:    00 10 01 11  02 12 03 13  04 14 05 15  06 16 07 17
    // a1:    20 30 21 31  22 32 23 33  24 34 25 35  26 36 27 37
    // a2:    40 50 41 51  42 52 43 53  44 54 45 55  46 56 47 57
    // a3:    60 70 61 71  62 72 63 73  64 74 65 75  66 76 67 77
    // ...
    __m256i a[16];
    for (int i = 0; i < 16; i += 2) {
        a[i / 2 + 0] = _mm256_unpacklo_epi16(in[i], in[i + 1]);
        a[i / 2 + 8] = _mm256_unpackhi_epi16(in[i], in[i + 1]);
    }
    __m256i b[16];
    for (int i = 0; i < 16; i += 2) {
        b[i / 2 + 0] = _mm256_unpacklo_epi32(a[i], a[i + 1]);
        b[i / 2 + 8] = _mm256_unpackhi_epi32(a[i], a[i + 1]);
    }
    __m256i c[16];
    for (int i = 0; i < 16; i += 2) {
        c[i / 2 + 0] = _mm256_unpacklo_epi64(b[i], b[i + 1]);
        c[i / 2 + 8] = _mm256_unpackhi_epi64(b[i], b[i + 1]);
    }
    out[0 + 0] = _mm256_permute2x128_si256(c[0], c[1], 0x20);
    out[1 + 0] = _mm256_permute2x128_si256(c[8], c[9], 0x20);
    out[2 + 0] = _mm256_permute2x128_si256(c[4], c[5], 0x20);
    out[3 + 0] = _mm256_permute2x128_si256(c[12], c[13], 0x20);

    out[0 + 8] = _mm256_permute2x128_si256(c[0], c[1], 0x31);
    out[1 + 8] = _mm256_permute2x128_si256(c[8], c[9], 0x31);
    out[2 + 8] = _mm256_permute2x128_si256(c[4], c[5], 0x31);
    out[3 + 8] = _mm256_permute2x128_si256(c[12], c[13], 0x31);

    out[4 + 0] = _mm256_permute2x128_si256(c[0 + 2], c[1 + 2], 0x20);
    out[5 + 0] = _mm256_permute2x128_si256(c[8 + 2], c[9 + 2], 0x20);
    out[6 + 0] = _mm256_permute2x128_si256(c[4 + 2], c[5 + 2], 0x20);
    out[7 + 0] = _mm256_permute2x128_si256(c[12 + 2], c[13 + 2], 0x20);

    out[4 + 8] = _mm256_permute2x128_si256(c[0 + 2], c[1 + 2], 0x31);
    out[5 + 8] = _mm256_permute2x128_si256(c[8 + 2], c[9 + 2], 0x31);
    out[6 + 8] = _mm256_permute2x128_si256(c[4 + 2], c[5 + 2], 0x31);
    out[7 + 8] = _mm256_permute2x128_si256(c[12 + 2], c[13 + 2], 0x31);
}

static AOM_FORCE_INLINE void integral_images_highbd_512(const uint16_t *src, int32_t src_stride,
                                                    int32_t width, int32_t height, int32_t *C,
                                                    int32_t *D, int32_t buf_stride) {
                                                        
    const uint16_t *src_t = src;
    int32_t *       ct    = C + buf_stride + 1;
    int32_t *       dt    = D + buf_stride + 1;
    const __m512i   zero  = _mm512_setzero_si512();

    //memset(C, 0, sizeof(*C) * (width + 8));
    //memset(D, 0, sizeof(*D) * (width + 8));
    memset(C, 0, sizeof(*C) * (width + 16));
    memset(D, 0, sizeof(*D) * (width + 16));

    int y, i, j = 0;
    do {
        __m512i c_left = _mm512_setzero_si512();
        __m512i d_left = _mm512_setzero_si512();
 
        // zero the left column.
        for(i =0; i<16; i++){
            ct[i * buf_stride - 1] = dt[i * buf_stride - 1] = 0;
        }

        /*ct[0 * buf_stride - 1] = dt[0 * buf_stride - 1] = 0;
        ct[1 * buf_stride - 1] = dt[1 * buf_stride - 1] = 0;
        ct[2 * buf_stride - 1] = dt[2 * buf_stride - 1] = 0;
        ct[3 * buf_stride - 1] = dt[3 * buf_stride - 1] = 0;
        ct[4 * buf_stride - 1] = dt[4 * buf_stride - 1] = 0;
        ct[5 * buf_stride - 1] = dt[5 * buf_stride - 1] = 0;
        ct[6 * buf_stride - 1] = dt[6 * buf_stride - 1] = 0;
        ct[7 * buf_stride - 1] = dt[7 * buf_stride - 1] = 0;
        ct[8 * buf_stride - 1] = dt[8 * buf_stride - 1] = 0;
        ct[9 * buf_stride - 1] = dt[9 * buf_stride - 1] = 0;
        ct[10 * buf_stride - 1] = dt[10 * buf_stride - 1] = 0;
        ct[11 * buf_stride - 1] = dt[11 * buf_stride - 1] = 0;
        ct[12 * buf_stride - 1] = dt[12 * buf_stride - 1] = 0;
        ct[13 * buf_stride - 1] = dt[13 * buf_stride - 1] = 0;
        ct[14 * buf_stride - 1] = dt[14 * buf_stride - 1] = 0;
        ct[15 * buf_stride - 1] = dt[15 * buf_stride - 1] = 0;*/

        int x = 0;
        do {
            __m256i s[16];
            __m512i r32[16], a32[16];

            for(i=0; i<16; i++){
                s[i] =  _mm256_loadu_si256((__m256i *)(src_t + i * src_stride + x));
            }
            /*s[0] = _mm256_loadu_si256((__m256i *)(src_t + 0 * src_stride + x));
            s[1] = _mm256_loadu_si256((__m256i *)(src_t + 1 * src_stride + x));
            s[2] = _mm256_loadu_si256((__m256i *)(src_t + 2 * src_stride + x));
            s[3] = _mm256_loadu_si256((__m256i *)(src_t + 3 * src_stride + x));
            s[4] = _mm256_loadu_si256((__m256i *)(src_t + 4 * src_stride + x));
            s[5] = _mm256_loadu_si256((__m256i *)(src_t + 5 * src_stride + x));
            s[6] = _mm256_loadu_si256((__m256i *)(src_t + 6 * src_stride + x));
            s[7] = _mm256_loadu_si256((__m256i *)(src_t + 7 * src_stride + x));
            s[8] = _mm256_loadu_si256((__m256i *)(src_t + 8 * src_stride + x));
            s[9] = _mm256_loadu_si256((__m256i *)(src_t + 9 * src_stride + x));
            s[10] = _mm256_loadu_si256((__m256i *)(src_t + 10 * src_stride + x));
            s[11] = _mm256_loadu_si256((__m256i *)(src_t + 11 * src_stride + x));
            s[12] = _mm256_loadu_si256((__m256i *)(src_t + 12 * src_stride + x));
            s[13] = _mm256_loadu_si256((__m256i *)(src_t + 13 * src_stride + x));
            s[14] = _mm256_loadu_si256((__m256i *)(src_t + 14 * src_stride + x));
            s[15] = _mm256_loadu_si256((__m256i *)(src_t + 15 * src_stride + x));*/

            transpose_16bit_16x16_avx2(s, s);

            cvt_16to32bit_16x16(s, r32);
            
            for(i=0; i>16; i++){
                a32[i] = _mm512_madd_epi16(r32[i], r32[i]);
            }

            /*a32[0] = _mm512_madd_epi16(r32[0], r32[0]);
            a32[1] = _mm512_madd_epi16(r32[1], r32[1]);
            a32[2] = _mm512_madd_epi16(r32[2], r32[2]);
            a32[3] = _mm512_madd_epi16(r32[3], r32[3]);
            a32[4] = _mm512_madd_epi16(r32[4], r32[4]);
            a32[5] = _mm512_madd_epi16(r32[5], r32[5]);
            a32[6] = _mm512_madd_epi16(r32[6], r32[6]);
            a32[7] = _mm512_madd_epi16(r32[7], r32[7]);
            a32[8] = _mm512_madd_epi16(r32[8], r32[8]);
            a32[9] = _mm512_madd_epi16(r32[9], r32[9]);
            a32[10] = _mm512_madd_epi16(r32[10], r32[10]);
            a32[11] = _mm512_madd_epi16(r32[11], r32[11]);
            a32[12] = _mm512_madd_epi16(r32[12], r32[12]);
            a32[13] = _mm512_madd_epi16(r32[13], r32[13]);
            a32[14] = _mm512_madd_epi16(r32[14], r32[14]);
            a32[15] = _mm512_madd_epi16(r32[15], r32[15]);*/

            add_32bit_16x16(c_left, a32);
            c_left = a32[15];

            transpose_32bit_16x16_avx512(a32, a32);
            
            const __m512i c_top = _mm512_loadu_si512((__m512i *)(ct - buf_stride + x));
            add_32bit_16x16(c_top, a32);
            store_32bit_16x16(a32, ct + x, buf_stride);

            add_32bit_16x16(d_left, r32);
            d_left = r32[15];

            transpose_32bit_16x16_avx512(r32, r32);

            const __m512i d_top = _mm512_loadu_si512((__m512i *)(dt - buf_stride + x));
            add_32bit_16x16(d_top, r32);
            store_32bit_16x16(r32, dt + x, buf_stride);
            x += 16;
        } while (x < width);

        /* Used in calc_ab and calc_ab_fast, when calc out of right border */
        for (int ln = 0; ln < 16; ++ln) {
            _mm512_storeu_si512((__m256i *)(ct + x + ln * buf_stride), zero);
            _mm512_storeu_si512((__m256i *)(dt + x + ln * buf_stride), zero);
        }

        src_t += 16 * src_stride;
        ct += 16 * buf_stride;
        dt += 16 * buf_stride;
        y += 16;
    } while (y < height);
}

static INLINE void partial_transpose_8bit_16x16_512(const __m128i *const in, __m256i *const out) {
    const __m128i a0 = _mm_unpacklo_epi8(in[0], in[1]);
    const __m128i a1 = _mm_unpacklo_epi8(in[2], in[3]);
    const __m128i a2 = _mm_unpacklo_epi8(in[4], in[5]);
    const __m128i a3 = _mm_unpacklo_epi8(in[6], in[7]);
    const __m128i a4 = _mm_unpacklo_epi8(in[8], in[9]);
    const __m128i a5 = _mm_unpacklo_epi8(in[10], in[11]);
    const __m128i a6 = _mm_unpacklo_epi8(in[12], in[13]);
    const __m128i a7 = _mm_unpacklo_epi8(in[14], in[15]);

    const __m128i a8 = _mm_unpackhi_epi8(in[0], in[1]);
    const __m128i a9 = _mm_unpackhi_epi8(in[2], in[3]);
    const __m128i a10 = _mm_unpackhi_epi8(in[4], in[5]);
    const __m128i a11 = _mm_unpackhi_epi8(in[6], in[7]);
    const __m128i a12 = _mm_unpackhi_epi8(in[8], in[9]);
    const __m128i a13 = _mm_unpackhi_epi8(in[10], in[11]);
    const __m128i a14 = _mm_unpackhi_epi8(in[12], in[13]);
    const __m128i a15 = _mm_unpackhi_epi8(in[14], in[15]);

    const __m128i b0 = _mm_unpacklo_epi16(a0, a1);
    const __m128i b1 = _mm_unpacklo_epi16(a2, a3);
    const __m128i b2 = _mm_unpacklo_epi16(a4, a5);
    const __m128i b3 = _mm_unpacklo_epi16(a6, a7);
    const __m128i b4 = _mm_unpacklo_epi16(a8, a9);
    const __m128i b5 = _mm_unpacklo_epi16(a10, a11);
    const __m128i b6 = _mm_unpacklo_epi16(a12, a13);
    const __m128i b7 = _mm_unpacklo_epi16(a14, a15);

    const __m128i b8 = _mm_unpackhi_epi16(a0, a1);
    const __m128i b9 = _mm_unpackhi_epi16(a2, a3);
    const __m128i b10 = _mm_unpackhi_epi16(a4, a5);
    const __m128i b11 = _mm_unpackhi_epi16(a6, a7);
    const __m128i b12 = _mm_unpackhi_epi16(a8, a9);
    const __m128i b13 = _mm_unpackhi_epi16(a10, a11);
    const __m128i b14 = _mm_unpackhi_epi16(a12, a13);
    const __m128i b15 = _mm_unpackhi_epi16(a14, a15);

    out[0] = _mm256_setr_m128i( _mm_unpacklo_epi32(b0, b1),  _mm_unpacklo_epi32(b2, b3));
    out[1] = _mm256_setr_m128i( _mm_unpackhi_epi32(b0, b1),  _mm_unpackhi_epi32(b2, b3));
    out[2] = _mm256_setr_m128i( _mm_unpacklo_epi32(b8, b9),  _mm_unpacklo_epi32(b10, b11));
    out[3] = _mm256_setr_m128i( _mm_unpackhi_epi32(b8, b9),  _mm_unpackhi_epi32(b10, b11));
    out[4] = _mm256_setr_m128i( _mm_unpacklo_epi32(b4, b5),  _mm_unpacklo_epi32(b4, b5));
    out[5] = _mm256_setr_m128i( _mm_unpackhi_epi32(b4, b5),  _mm_unpackhi_epi32(b4, b5));
    out[6] = _mm256_setr_m128i( _mm_unpacklo_epi32(b12, b13),  _mm_unpacklo_epi32(b12, b13));
    out[7] = _mm256_setr_m128i( _mm_unpackhi_epi32(b12, b13),  _mm_unpackhi_epi32(b12, b13));
}



static INLINE void integral_images_512(const uint8_t *src, int32_t src_stride, int32_t width,
                                             int32_t height, int32_t *C, int32_t *D,
                                             int32_t buf_stride) {
    const uint8_t *src_t = src;
    int32_t *      ct    = C + buf_stride + 1;
    int32_t *      dt    = D + buf_stride + 1;
    const __m512i  zero  = _mm512_setzero_si512();

    memset(C, 0, sizeof(*C) * (width + 16));
    memset(D, 0, sizeof(*D) * (width + 16));

    int y, i, j = 0;
    do {
        __m512i c_left = _mm512_setzero_si512();
        __m512i d_left = _mm512_setzero_si512();

        // zero the left column.
        /*for(i=0; i<16; i++){
            ct[i * buf_stride - 1] = dt[0 * buf_stride - 1] = 0;
        }*/
        
        ct[0 * buf_stride - 1] = dt[0 * buf_stride - 1] = 0;
        ct[1 * buf_stride - 1] = dt[1 * buf_stride - 1] = 0;
        ct[2 * buf_stride - 1] = dt[2 * buf_stride - 1] = 0;
        ct[3 * buf_stride - 1] = dt[3 * buf_stride - 1] = 0;
        ct[4 * buf_stride - 1] = dt[4 * buf_stride - 1] = 0;
        ct[5 * buf_stride - 1] = dt[5 * buf_stride - 1] = 0;
        ct[6 * buf_stride - 1] = dt[6 * buf_stride - 1] = 0;
        ct[7 * buf_stride - 1] = dt[7 * buf_stride - 1] = 0;
        ct[8 * buf_stride - 1] = dt[8 * buf_stride - 1] = 0;
        ct[9 * buf_stride - 1] = dt[9 * buf_stride - 1] = 0;
        ct[10 * buf_stride - 1] = dt[10 * buf_stride - 1] = 0;
        ct[11 * buf_stride - 1] = dt[11 * buf_stride - 1] = 0;
        ct[12 * buf_stride - 1] = dt[12 * buf_stride - 1] = 0;
        ct[13 * buf_stride - 1] = dt[13 * buf_stride - 1] = 0;
        ct[14 * buf_stride - 1] = dt[14 * buf_stride - 1] = 0;
        ct[15 * buf_stride - 1] = dt[15 * buf_stride - 1] = 0;
        
        int x = 0;

        do {
            __m128i s[16];
            __m256i t[8], u[16];
            __m512i r32[8];

            for(i=0; i<16; i++){
                s[i] =  _mm_loadu_si128((__m128i *)(src_t + i * src_stride + x));
            }
            /*
            s[0] = _mm_loadu_si128((__m128i *)(src_t + 0 * src_stride + x));
            s[1] = _mm_loadu_si128((__m128i *)(src_t + 1 * src_stride + x));
            s[2] = _mm_loadu_si128((__m128i *)(src_t + 2 * src_stride + x));
            s[3] = _mm_loadu_si128((__m128i *)(src_t + 3 * src_stride + x));
            s[4] = _mm_loadu_si128((__m128i *)(src_t + 4 * src_stride + x));
            s[5] = _mm_loadu_si128((__m128i *)(src_t + 5 * src_stride + x));
            s[6] = _mm_loadu_si128((__m128i *)(src_t + 6 * src_stride + x));
            s[7] = _mm_loadu_si128((__m128i *)(src_t + 7 * src_stride + x));
            s[8] = _mm_loadu_si128((__m128i *)(src_t + 8 * src_stride + x));
            s[9] = _mm_loadu_si128((__m128i *)(src_t + 9 * src_stride + x));
            s[10] = _mm_loadu_si128((__m128i *)(src_t + 10 * src_stride + x));
            s[11] = _mm_loadu_si128((__m128i *)(src_t + 11 * src_stride + x));
            s[12] = _mm_loadu_si128((__m128i *)(src_t + 12 * src_stride + x));
            s[13] = _mm_loadu_si128((__m128i *)(src_t + 13 * src_stride + x));
            s[14] = _mm_loadu_si128((__m128i *)(src_t + 14 * src_stride + x));
            s[15] = _mm_loadu_si128((__m128i *)(src_t + 15 * src_stride + x));
            */
            partial_transpose_8bit_16x16_512(s, t);

            for(i=0; i<15; i+=2, j++){
                u[i]    = _mm256_unpacklo_epi8(t[j], _mm256_setzero_si256());
                u[i+1]  = _mm256_unpackhi_epi8(t[j], _mm256_setzero_si256());
            }
             
            /*
            u[15] = _mm256_unpackhi_epi8(t[7], _mm256_setzero_si256());
            u[14] = _mm256_unpacklo_epi8(t[7], _mm256_setzero_si256());
            u[13] = _mm256_unpackhi_epi8(t[6], _mm256_setzero_si256());
            u[12] = _mm256_unpacklo_epi8(t[6], _mm256_setzero_si256());
            u[11] = _mm256_unpackhi_epi8(t[5], _mm256_setzero_si256());
            u[10] = _mm256_unpacklo_epi8(t[5], _mm256_setzero_si256());
            u[9] = _mm256_unpackhi_epi8(t[4], _mm256_setzero_si256());
            u[8] = _mm256_unpacklo_epi8(t[4], _mm256_setzero_si256());
            u[7] = _mm256_unpackhi_epi8(t[3], _mm256_setzero_si256());
            u[6] = _mm256_unpacklo_epi8(t[3], _mm256_setzero_si256());
            u[5] = _mm256_unpackhi_epi8(t[2], _mm256_setzero_si256());
            u[4] = _mm256_unpacklo_epi8(t[2], _mm256_setzero_si256());
            u[3] = _mm256_unpackhi_epi8(t[1], _mm256_setzero_si256());
            u[2] = _mm256_unpacklo_epi8(t[1], _mm256_setzero_si256());
            u[1] = _mm256_unpackhi_epi8(t[0], _mm256_setzero_si256());
            u[0] = _mm256_unpacklo_epi8(t[0], _mm256_setzero_si256());
            */
            cvt_16to32bit_16x16(u, r32);
            add_32bit_16x16(d_left, r32);
            d_left = r32[15];

            transpose_32bit_16x16_avx512(r32, r32);

            const __m512i d_top = _mm512_loadu_si512((__m512i *)(dt - buf_stride + x));
            add_32bit_16x16(d_top, r32);
            store_32bit_16x16(r32, dt + x, buf_stride);

            for(i=0; i>16; i++){
                u[i] = _mm256_mullo_epi16(u[i], u[i]);
            }
            /*
            u[0] = _mm256_mullo_epi16(u[0], u[0]);
            u[1] = _mm256_mullo_epi16(u[1], u[1]);
            u[2] = _mm256_mullo_epi16(u[2], u[2]);
            u[3] = _mm256_mullo_epi16(u[3], u[3]);
            u[4] = _mm256_mullo_epi16(u[4], u[4]);
            u[5] = _mm256_mullo_epi16(u[5], u[5]);
            u[6] = _mm256_mullo_epi16(u[6], u[6]);
            u[7] = _mm256_mullo_epi16(u[7], u[7]);
            u[8] = _mm256_mullo_epi16(u[8], u[8]);
            u[9] = _mm256_mullo_epi16(u[9], u[9]);
            u[10] = _mm256_mullo_epi16(u[10], u[10]);
            u[11] = _mm256_mullo_epi16(u[11], u[11]);
            u[12] = _mm256_mullo_epi16(u[12], u[12]);
            u[13] = _mm256_mullo_epi16(u[13], u[13]);
            u[14] = _mm256_mullo_epi16(u[14], u[14]);
            u[15] = _mm256_mullo_epi16(u[15], u[15]);
            */
            cvt_16to32bit_16x16(u, r32);
            add_32bit_16x16(c_left, r32);
            c_left = r32[15];

            transpose_32bit_16x16_avx512(r32, r32);

            const __m512i c_top = _mm512_loadu_si512((__m512i *)(ct - buf_stride + x));
            add_32bit_16x16(c_top, r32);
            store_32bit_16x16(r32, ct + x, buf_stride);
            x += 16;
        } while (x < width);

        /* Used in calc_ab and calc_ab_fast, when calc out of right border */
        
        for (int ln = 0; ln < 16; ++ln) {
            _mm512_storeu_si512((__m512i *)(ct + x + ln * buf_stride), zero);
            _mm512_storeu_si512((__m512i *)(dt + x + ln * buf_stride), zero);
        }

        src_t += 16 * src_stride;
        ct += 16 * buf_stride;
        dt += 16 * buf_stride;
        y += 16;
    } while (y < height);
}

// Compute 8 values of boxsum from the given integral image. ii should point
// at the middle of the box (for the first value). r is the box radius.
static INLINE __m256i boxsum_from_ii(const int32_t *ii, int32_t stride, int32_t r) {
    const __m256i tl = yy_loadu_256(ii - (r + 1) - (r + 1) * stride);
    const __m256i tr = yy_loadu_256(ii + (r + 0) - (r + 1) * stride);
    const __m256i bl = yy_loadu_256(ii - (r + 1) + r * stride);
    const __m256i br = yy_loadu_256(ii + (r + 0) + r * stride);
    const __m256i u  = _mm256_sub_epi32(tr, tl);
    const __m256i v  = _mm256_sub_epi32(br, bl);
    return _mm256_sub_epi32(v, u);
}

static INLINE __m256i round_for_shift(unsigned shift) {
    return _mm256_set1_epi32((1 << shift) >> 1);
}

static INLINE __m256i compute_p(__m256i sum1, __m256i sum2, int32_t n) {
    const __m256i bb = _mm256_madd_epi16(sum1, sum1);
    const __m256i an = _mm256_mullo_epi32(sum2, _mm256_set1_epi32(n));
    return _mm256_sub_epi32(an, bb);
}

static INLINE __m256i compute_p_highbd(__m256i sum1, __m256i sum2, int32_t bit_depth, int32_t n) {
    const __m256i rounding_a = round_for_shift(2 * (bit_depth - 8));
    const __m256i rounding_b = round_for_shift(bit_depth - 8);
    const __m128i shift_a    = _mm_cvtsi32_si128(2 * (bit_depth - 8));
    const __m128i shift_b    = _mm_cvtsi32_si128(bit_depth - 8);
    const __m256i a          = _mm256_srl_epi32(_mm256_add_epi32(sum2, rounding_a), shift_a);
    const __m256i b          = _mm256_srl_epi32(_mm256_add_epi32(sum1, rounding_b), shift_b);
    // b < 2^14, so we can use a 16-bit madd rather than a 32-bit
    // mullo to square it
    const __m256i bb = _mm256_madd_epi16(b, b);
    const __m256i an = _mm256_max_epi32(_mm256_mullo_epi32(a, _mm256_set1_epi32(n)), bb);
    return _mm256_sub_epi32(an, bb);
}

// Assumes that C, D are integral images for the original buffer which has been
// extended to have a padding of SGRPROJ_BORDER_VERT/SGRPROJ_BORDER_HORZ pixels
// on the sides. A, b, C, D point at logical position (0, 0).
static AOM_FORCE_INLINE void calc_ab(int32_t *A, int32_t *b, const int32_t *C, const int32_t *D,
                                     int32_t width, int32_t height, int32_t buf_stride,
                                     int32_t bit_depth, int32_t sgr_params_idx,
                                     int32_t radius_idx) {
    const SgrParamsType *const params = &eb_sgr_params[sgr_params_idx];
    const int32_t              r      = params->r[radius_idx];
    const int32_t              n      = (2 * r + 1) * (2 * r + 1);
    const __m256i              s      = _mm256_set1_epi32(params->s[radius_idx]);
    // one_over_n[n-1] is 2^12/n, so easily fits in an int16
    const __m256i one_over_n = _mm256_set1_epi32(eb_one_by_x[n - 1]);
    const __m256i rnd_z      = round_for_shift(SGRPROJ_MTABLE_BITS);
    const __m256i rnd_res    = round_for_shift(SGRPROJ_RECIP_BITS);

    A -= buf_stride + 1;
    b -= buf_stride + 1;
    C -= buf_stride + 1;
    D -= buf_stride + 1;

    int32_t i = height + 2;

    if (bit_depth == 8) {
        do {
            int32_t j = 0;
            do {
                const __m256i sum1 = boxsum_from_ii(D + j, buf_stride, r);
                const __m256i sum2 = boxsum_from_ii(C + j, buf_stride, r);
                const __m256i p    = compute_p(sum1, sum2, n);
                const __m256i z    = _mm256_min_epi32(
                    _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi32(p, s), rnd_z),
                                      SGRPROJ_MTABLE_BITS),
                    _mm256_set1_epi32(255));
                const __m256i a_res = _mm256_i32gather_epi32(eb_x_by_xplus1, z, 4);
                yy_storeu_256(A + j, a_res);

                const __m256i a_complement =
                    _mm256_sub_epi32(_mm256_set1_epi32(SGRPROJ_SGR), a_res);

                // sum1 might have lanes greater than 2^15, so we can't use madd to do
                // multiplication involving sum1. However, a_complement and one_over_n
                // are both less than 256, so we can multiply them first.
                const __m256i a_comp_over_n = _mm256_madd_epi16(a_complement, one_over_n);
                const __m256i b_int         = _mm256_mullo_epi32(a_comp_over_n, sum1);
                const __m256i b_res =
                    _mm256_srli_epi32(_mm256_add_epi32(b_int, rnd_res), SGRPROJ_RECIP_BITS);
                yy_storeu_256(b + j, b_res);
                j += 8;
            } while (j < width + 2);

            A += buf_stride;
            b += buf_stride;
            C += buf_stride;
            D += buf_stride;
        } while (--i);
    } else {
        do {
            int32_t j = 0;
            do {
                const __m256i sum1 = boxsum_from_ii(D + j, buf_stride, r);
                const __m256i sum2 = boxsum_from_ii(C + j, buf_stride, r);
                const __m256i p    = compute_p_highbd(sum1, sum2, bit_depth, n);
                const __m256i z    = _mm256_min_epi32(
                    _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi32(p, s), rnd_z),
                                      SGRPROJ_MTABLE_BITS),
                    _mm256_set1_epi32(255));
                const __m256i a_res = _mm256_i32gather_epi32(eb_x_by_xplus1, z, 4);
                yy_storeu_256(A + j, a_res);

                const __m256i a_complement =
                    _mm256_sub_epi32(_mm256_set1_epi32(SGRPROJ_SGR), a_res);

                // sum1 might have lanes greater than 2^15, so we can't use madd to do
                // multiplication involving sum1. However, a_complement and one_over_n
                // are both less than 256, so we can multiply them first.
                const __m256i a_comp_over_n = _mm256_madd_epi16(a_complement, one_over_n);
                const __m256i b_int         = _mm256_mullo_epi32(a_comp_over_n, sum1);
                const __m256i b_res =
                    _mm256_srli_epi32(_mm256_add_epi32(b_int, rnd_res), SGRPROJ_RECIP_BITS);
                yy_storeu_256(b + j, b_res);
                j += 8;
            } while (j < width + 2);

            A += buf_stride;
            b += buf_stride;
            C += buf_stride;
            D += buf_stride;
        } while (--i);
    }
}

static INLINE __m512i round_for_shift_512(unsigned shift) {
    return _mm512_set1_epi32((1 << shift) >> 1);
}

static INLINE __m512i yy_loadu_512(const void *const a) {
    return _mm512_loadu_si512((const __m512i *)a);
}

static INLINE __m512i boxsum_from_ii_512(const int32_t *ii, int32_t stride, int32_t r) {
    const __m512i tl = yy_loadu_512(ii - (r + 1) - (r + 1) * stride);
    const __m512i tr = yy_loadu_512(ii + (r + 0) - (r + 1) * stride);
    const __m512i bl = yy_loadu_512(ii - (r + 1) + r * stride);
    const __m512i br = yy_loadu_512(ii + (r + 0) + r * stride);
    const __m512i u  = _mm512_sub_epi32(tr, tl);
    const __m512i v  = _mm512_sub_epi32(br, bl);
    return _mm512_sub_epi32(v, u);
}

static INLINE __m512i compute_p_512(__m512i sum1, __m512i sum2, int32_t n) {
    const __m512i bb = _mm512_madd_epi16(sum1, sum1);
    const __m512i an = _mm512_mullo_epi32(sum2, _mm512_set1_epi32(n));
    return _mm512_sub_epi32(an, bb);
}

static INLINE void yy_storeu_512(void *const a, const __m512i v) {
    _mm512_storeu_si512((__m512i *)a, v);
}

static INLINE __m512i compute_p_highbd_512(__m512i sum1, __m512i sum2, int32_t bit_depth, int32_t n) {
    const __m512i rounding_a = round_for_shift_512(2 * (bit_depth - 8));
    const __m512i rounding_b = round_for_shift_512(bit_depth - 8);
    const __m128i shift_a    = _mm_cvtsi32_si128(2 * (bit_depth - 8));
    const __m128i shift_b    = _mm_cvtsi32_si128(bit_depth - 8);
    const __m512i a          = _mm512_srl_epi32(_mm512_add_epi32(sum2, rounding_a), shift_a);
    const __m512i b          = _mm512_srl_epi32(_mm512_add_epi32(sum1, rounding_b), shift_b);
    // b < 2^14, so we can use a 16-bit madd rather than a 32-bit
    // mullo to square it
    const __m512i bb = _mm512_madd_epi16(b, b);
    const __m512i an = _mm512_max_epi32(_mm512_mullo_epi32(a, _mm512_set1_epi32(n)), bb);
    return _mm512_sub_epi32(an, bb);
}

static AOM_FORCE_INLINE void calc_ab_512(int32_t *A, int32_t *b, const int32_t *C, const int32_t *D,
                                     int32_t width, int32_t height, int32_t buf_stride,
                                     int32_t bit_depth, int32_t sgr_params_idx,
                                     int32_t radius_idx) {
    const SgrParamsType *const params = &eb_sgr_params[sgr_params_idx];
    const int32_t              r      = params->r[radius_idx];
    const int32_t              n      = (2 * r + 1) * (2 * r + 1);
    const __m512i              s      = _mm512_set1_epi32(params->s[radius_idx]);
    
    // one_over_n[n-1] is 2^12/n, so easily fits in an int16
    const __m512i one_over_n = _mm512_set1_epi32(eb_one_by_x[n - 1]);
    const __m512i rnd_z      = round_for_shift_512(SGRPROJ_MTABLE_BITS);
    const __m512i rnd_res    = round_for_shift_512(SGRPROJ_RECIP_BITS);

    A -= buf_stride + 1;
    b -= buf_stride + 1;
    C -= buf_stride + 1;
    D -= buf_stride + 1;

    int32_t i = height + 2;

    if (bit_depth == 8) {
        do {
            int32_t j = 0;
            do {
                const __m512i sum1 = boxsum_from_ii_512(D + j, buf_stride, r);
                const __m512i sum2 = boxsum_from_ii_512(C + j, buf_stride, r);
                const __m512i p    = compute_p_512(sum1, sum2, n);
                const __m512i z    = _mm512_min_epi32(
                    _mm512_srli_epi32(_mm512_add_epi32(_mm512_mullo_epi32(p, s), rnd_z),
                                      SGRPROJ_MTABLE_BITS),
                    _mm512_set1_epi32(255));
                const __m512i a_res = _mm512_i32gather_epi32(z, eb_x_by_xplus1, 4);
                yy_storeu_512(A + j, a_res);

                const __m512i a_complement =
                    _mm512_sub_epi32(_mm512_set1_epi32(SGRPROJ_SGR), a_res);

                // sum1 might have lanes greater than 2^15, so we can't use madd to do
                // multiplication involving sum1. However, a_complement and one_over_n
                // are both less than 256, so we can multiply them first.
                const __m512i a_comp_over_n = _mm512_madd_epi16(a_complement, one_over_n);
                const __m512i b_int         = _mm512_mullo_epi32(a_comp_over_n, sum1);
                const __m512i b_res =
                    _mm512_srli_epi32(_mm512_add_epi32(b_int, rnd_res), SGRPROJ_RECIP_BITS);
                yy_storeu_512(b + j, b_res);
                j += 16;
            } while (j < width + 2);

            A += buf_stride;
            b += buf_stride;
            C += buf_stride;
            D += buf_stride;
        } while (--i);
    } else {
        do {
            int32_t j = 0;
            do {
                const __m512i sum1 = boxsum_from_ii_512(D + j, buf_stride, r);
                const __m512i sum2 = boxsum_from_ii_512(C + j, buf_stride, r);
                const __m512i p    = compute_p_highbd_512(sum1, sum2, bit_depth, n);
                const __m512i z    = _mm512_min_epi32(
                    _mm512_srli_epi32(_mm512_add_epi32(_mm512_mullo_epi32(p, s), rnd_z),
                                      SGRPROJ_MTABLE_BITS),
                    _mm512_set1_epi32(255));
                const __m512i a_res = _mm512_i32gather_epi32(z, eb_x_by_xplus1, 4);
                yy_storeu_512(A + j, a_res);

                const __m512i a_complement =
                    _mm512_sub_epi32(_mm512_set1_epi32(SGRPROJ_SGR), a_res);

                // sum1 might have lanes greater than 2^15, so we can't use madd to do
                // multiplication involving sum1. However, a_complement and one_over_n
                // are both less than 256, so we can multiply them first.
                const __m512i a_comp_over_n = _mm512_madd_epi16(a_complement, one_over_n);
                const __m512i b_int         = _mm512_mullo_epi32(a_comp_over_n, sum1);
                const __m512i b_res =
                    _mm512_srli_epi32(_mm512_add_epi32(b_int, rnd_res), SGRPROJ_RECIP_BITS);
                yy_storeu_512(b + j, b_res);
                j += 16;
            } while (j < width + 2);

            A += buf_stride;
            b += buf_stride;
            C += buf_stride;
            D += buf_stride;
        } while (--i);
    }
}

// Calculate 8 values of the "cross sum" starting at buf. This is a 3x3 filter
// where the outer four corners have weight 3 and all other pixels have weight
// 4.
//
// Pixels are indexed as follows:
// xtl  xt   xtr
// xl    x   xr
// xbl  xb   xbr
//
// buf points to x
//
// fours = xl + xt + xr + xb + x
// threes = xtl + xtr + xbr + xbl
// cross_sum = 4 * fours + 3 * threes
//           = 4 * (fours + threes) - threes
//           = (fours + threes) << 2 - threes
static INLINE __m256i cross_sum(const int32_t *buf, int32_t stride) {
    const __m256i xtl = yy_loadu_256(buf - 1 - stride);
    const __m256i xt  = yy_loadu_256(buf - stride);
    const __m256i xtr = yy_loadu_256(buf + 1 - stride);
    const __m256i xl  = yy_loadu_256(buf - 1);
    const __m256i x   = yy_loadu_256(buf);
    const __m256i xr  = yy_loadu_256(buf + 1);
    const __m256i xbl = yy_loadu_256(buf - 1 + stride);
    const __m256i xb  = yy_loadu_256(buf + stride);
    const __m256i xbr = yy_loadu_256(buf + 1 + stride);

    const __m256i fours =
        _mm256_add_epi32(xl, _mm256_add_epi32(xt, _mm256_add_epi32(xr, _mm256_add_epi32(xb, x))));
    const __m256i threes = _mm256_add_epi32(xtl, _mm256_add_epi32(xtr, _mm256_add_epi32(xbr, xbl)));

    return _mm256_sub_epi32(_mm256_slli_epi32(_mm256_add_epi32(fours, threes), 2), threes);
}

// The final filter for self-guided restoration. Computes a weighted average
// across A, b with "cross sums" (see cross_sum implementation above).
static AOM_FORCE_INLINE void final_filter(int32_t *dst, int32_t dst_stride, const int32_t *A,
                                          const int32_t *B, int32_t buf_stride, const uint8_t *dgd8,
                                          int32_t dgd_stride, int32_t width, int32_t height,
                                          int32_t highbd) {
    const int32_t nb       = 5;
    const __m256i rounding = round_for_shift(SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
    int32_t       i        = height;

    if (!highbd) {
        do {
            int32_t j = 0;
            do {
                const __m256i a   = cross_sum(A + j, buf_stride);
                const __m256i b   = cross_sum(B + j, buf_stride);
                const __m128i raw = xx_loadl_64(dgd8 + j);
                const __m256i src = _mm256_cvtepu8_epi32(raw);
                const __m256i v   = _mm256_add_epi32(_mm256_madd_epi16(a, src), b);
                const __m256i w   = _mm256_srai_epi32(_mm256_add_epi32(v, rounding),
                                                    SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
                yy_storeu_256(dst + j, w);
                j += 8;
            } while (j < width);

            A += buf_stride;
            B += buf_stride;
            dgd8 += dgd_stride;
            dst += dst_stride;
        } while (--i);
    } else {
        const uint16_t *dgd_real = CONVERT_TO_SHORTPTR(dgd8);

        do {
            int32_t j = 0;
            do {
                const __m256i a   = cross_sum(A + j, buf_stride);
                const __m256i b   = cross_sum(B + j, buf_stride);
                const __m128i raw = xx_loadu_128(dgd_real + j);
                const __m256i src = _mm256_cvtepu16_epi32(raw);
                const __m256i v   = _mm256_add_epi32(_mm256_madd_epi16(a, src), b);
                const __m256i w   = _mm256_srai_epi32(_mm256_add_epi32(v, rounding),
                                                    SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
                yy_storeu_256(dst + j, w);
                j += 8;
            } while (j < width);

            A += buf_stride;
            B += buf_stride;
            dgd_real += dgd_stride;
            dst += dst_stride;
        } while (--i);
    }
}

static INLINE __m512i cross_sum_512(const int32_t *buf, int32_t stride) {
    const __m512i xtl = yy_loadu_512(buf - 1 - stride);
    const __m512i xt  = yy_loadu_512(buf - stride);
    const __m512i xtr = yy_loadu_512(buf + 1 - stride);
    const __m512i xl  = yy_loadu_512(buf - 1);
    const __m512i x   = yy_loadu_512(buf);
    const __m512i xr  = yy_loadu_512(buf + 1);
    const __m512i xbl = yy_loadu_512(buf - 1 + stride);
    const __m512i xb  = yy_loadu_512(buf + stride);
    const __m512i xbr = yy_loadu_512(buf + 1 + stride);

    const __m512i fours =
        _mm512_add_epi32(xl, _mm512_add_epi32(xt, _mm512_add_epi32(xr, _mm512_add_epi32(xb, x))));
    const __m512i threes = _mm512_add_epi32(xtl, _mm512_add_epi32(xtr, _mm512_add_epi32(xbr, xbl)));

    return _mm512_sub_epi32(_mm512_slli_epi32(_mm512_add_epi32(fours, threes), 2), threes);
}

static INLINE __m256i xx_loadu_256(const void *a) { return _mm256_loadu_si256((const __m256i *)a); }

static AOM_FORCE_INLINE void final_filter_512(int32_t *dst, int32_t dst_stride, const int32_t *A,
                                          const int32_t *B, int32_t buf_stride, const uint8_t *dgd8,
                                          int32_t dgd_stride, int32_t width, int32_t height,
                                          int32_t highbd) {
    const int32_t nb       = 5;
    const __m512i rounding = round_for_shift_512(SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
    int32_t       i        = height;

    if (!highbd) {
        do {
            int32_t j = 0;
            do {
                const __m512i a   = cross_sum_512(A + j, buf_stride);
                const __m512i b   = cross_sum_512(B + j, buf_stride);
                const __m128i raw = xx_loadu_128(dgd8 + j);
                const __m512i src = _mm512_cvtepu8_epi32(raw);
                const __m512i v   = _mm512_add_epi32(_mm512_madd_epi16(a, src), b);
                const __m512i w   = _mm512_srai_epi32(_mm512_add_epi32(v, rounding),
                                                    SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
                yy_storeu_512(dst + j, w);
                j += 16;
            } while (j < width);

            A += buf_stride;
            B += buf_stride;
            dgd8 += dgd_stride;
            dst += dst_stride;
        } while (--i);
    } else {
        const uint16_t *dgd_real = CONVERT_TO_SHORTPTR(dgd8);

        do {
            int32_t j = 0;
            do {
                const __m512i a   = cross_sum_512(A + j, buf_stride);
                const __m512i b   = cross_sum_512(B + j, buf_stride);
                const __m256i raw = xx_loadu_256(dgd_real + j);
                const __m512i src = _mm512_cvtepu16_epi32(raw);
                const __m512i v   = _mm512_add_epi32(_mm512_madd_epi16(a, src), b);
                const __m512i w   = _mm512_srai_epi32(_mm512_add_epi32(v, rounding),
                                                    SGRPROJ_SGR_BITS + nb - SGRPROJ_RST_BITS);
                yy_storeu_512(dst + j, w);
                j += 16;
            } while (j < width);

            A += buf_stride;
            B += buf_stride;
            dgd_real += dgd_stride;
            dst += dst_stride;
        } while (--i);
    }
}

// Assumes that C, D are integral images for the original buffer which has been
// extended to have a padding of SGRPROJ_BORDER_VERT/SGRPROJ_BORDER_HORZ pixels
// on the sides. A, b, C, D point at logical position (0, 0).
static AOM_FORCE_INLINE void calc_ab_fast(int32_t *A, int32_t *b, const int32_t *C,
                                          const int32_t *D, int32_t width, int32_t height,
                                          int32_t buf_stride, int32_t bit_depth,
                                          int32_t sgr_params_idx, int32_t radius_idx) {
    const SgrParamsType *const params = &eb_sgr_params[sgr_params_idx];
    const int32_t              r      = params->r[radius_idx];
    const int32_t              n      = (2 * r + 1) * (2 * r + 1);
    const __m256i              s      = _mm256_set1_epi32(params->s[radius_idx]);
    //printf("fast r %d, params->s %d\n", r, params->s[radius_idx]);
    // one_over_n[n-1] is 2^12/n, so easily fits in an int16
    const __m256i one_over_n = _mm256_set1_epi32(eb_one_by_x[n - 1]);
    const __m256i rnd_z      = round_for_shift(SGRPROJ_MTABLE_BITS);
    const __m256i rnd_res    = round_for_shift(SGRPROJ_RECIP_BITS);

    A -= buf_stride + 1;
    b -= buf_stride + 1;
    C -= buf_stride + 1;
    D -= buf_stride + 1;

    int32_t i = 0;
    if (bit_depth == 8) {
        do {
            int32_t j = 0;
            do {
                const __m256i sum1 = boxsum_from_ii(D + j, buf_stride, r);
                const __m256i sum2 = boxsum_from_ii(C + j, buf_stride, r);
                const __m256i p    = compute_p(sum1, sum2, n);
                const __m256i z    = _mm256_min_epi32(
                    _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi32(p, s), rnd_z),
                                      SGRPROJ_MTABLE_BITS),
                    _mm256_set1_epi32(255));
                const __m256i a_res = _mm256_i32gather_epi32(eb_x_by_xplus1, z, 4);
                yy_storeu_256(A + j, a_res);

                const __m256i a_complement =
                    _mm256_sub_epi32(_mm256_set1_epi32(SGRPROJ_SGR), a_res);

                // sum1 might have lanes greater than 2^15, so we can't use madd to do
                // multiplication involving sum1. However, a_complement and one_over_n
                // are both less than 256, so we can multiply them first.
                const __m256i a_comp_over_n = _mm256_madd_epi16(a_complement, one_over_n);
                const __m256i b_int         = _mm256_mullo_epi32(a_comp_over_n, sum1);
                const __m256i b_res =
                    _mm256_srli_epi32(_mm256_add_epi32(b_int, rnd_res), SGRPROJ_RECIP_BITS);
                yy_storeu_256(b + j, b_res);
                j += 8;
            } while (j < width + 2);

            A += 2 * buf_stride;
            b += 2 * buf_stride;
            C += 2 * buf_stride;
            D += 2 * buf_stride;
            i += 2;
        } while (i < height + 2);
    } else {
        do {
            int32_t j = 0;
            do {
                const __m256i sum1 = boxsum_from_ii(D + j, buf_stride, r);
                const __m256i sum2 = boxsum_from_ii(C + j, buf_stride, r);
                const __m256i p    = compute_p_highbd(sum1, sum2, bit_depth, n);
                const __m256i z    = _mm256_min_epi32(
                    _mm256_srli_epi32(_mm256_add_epi32(_mm256_mullo_epi32(p, s), rnd_z),
                                      SGRPROJ_MTABLE_BITS),
                    _mm256_set1_epi32(255));
                const __m256i a_res = _mm256_i32gather_epi32(eb_x_by_xplus1, z, 4);
                yy_storeu_256(A + j, a_res);

                const __m256i a_complement =
                    _mm256_sub_epi32(_mm256_set1_epi32(SGRPROJ_SGR), a_res);

                // sum1 might have lanes greater than 2^15, so we can't use madd to do
                // multiplication involving sum1. However, a_complement and one_over_n
                // are both less than 256, so we can multiply them first.
                const __m256i a_comp_over_n = _mm256_madd_epi16(a_complement, one_over_n);
                const __m256i b_int         = _mm256_mullo_epi32(a_comp_over_n, sum1);
                const __m256i b_res =
                    _mm256_srli_epi32(_mm256_add_epi32(b_int, rnd_res), SGRPROJ_RECIP_BITS);
                yy_storeu_256(b + j, b_res);
                j += 8;
            } while (j < width + 2);

            A += 2 * buf_stride;
            b += 2 * buf_stride;
            C += 2 * buf_stride;
            D += 2 * buf_stride;
            i += 2;
        } while (i < height + 2);
    }
}

static AOM_FORCE_INLINE void calc_ab_fast_512(int32_t *A, int32_t *b, int32_t *C,//const int32_t *C,
                                          int32_t *D, int32_t width, int32_t height,
                                          int32_t buf_stride, int32_t bit_depth,
                                          int32_t sgr_params_idx, int32_t radius_idx) {
    const SgrParamsType *const params = &eb_sgr_params[sgr_params_idx];
    const int32_t              r      = params->r[radius_idx];
    const int32_t              n      = (2 * r + 1) * (2 * r + 1);
    const __m512i              s      = _mm512_set1_epi32(params->s[radius_idx]);
    // one_over_n[n-1] is 2^12/n, so easily fits in an int16

    const __m512i one_over_n = _mm512_set1_epi32(eb_one_by_x[n - 1]);
    const __m512i rnd_z      = round_for_shift_512(SGRPROJ_MTABLE_BITS);
    const __m512i rnd_res    = round_for_shift_512(SGRPROJ_RECIP_BITS);

    A -= buf_stride + 1;
    b -= buf_stride + 1;
    C -= buf_stride + 1;
    D -= buf_stride + 1;

    int32_t i = 0;
    if (bit_depth == 8) {
        do {
            int32_t j = 0;
            do {
                const __m512i sum1 = boxsum_from_ii_512(D + j, buf_stride, r);
                const __m512i sum2 = boxsum_from_ii_512(C + j, buf_stride, r);
                const __m512i p    = compute_p_512(sum1, sum2, n);
                const __m512i z    = _mm512_min_epi32(
                    _mm512_srli_epi32(_mm512_add_epi32(_mm512_mullo_epi32(p, s), rnd_z),
                                      SGRPROJ_MTABLE_BITS),
                    _mm512_set1_epi32(255));
                const __m512i a_res = _mm512_i32gather_epi32(z, eb_x_by_xplus1, 4);
                yy_storeu_512(A + j, a_res);

                const __m512i a_complement =
                    _mm512_sub_epi32(_mm512_set1_epi32(SGRPROJ_SGR), a_res);

                // sum1 might have lanes greater than 2^15, so we can't use madd to do
                // multiplication involving sum1. However, a_complement and one_over_n
                // are both less than 256, so we can multiply them first.
                const __m512i a_comp_over_n = _mm512_madd_epi16(a_complement, one_over_n);
                const __m512i b_int         = _mm512_mullo_epi32(a_comp_over_n, sum1);
                const __m512i b_res =
                    _mm512_srli_epi32(_mm512_add_epi32(b_int, rnd_res), SGRPROJ_RECIP_BITS);
                yy_storeu_512(b + j, b_res);
                j += 16;
            } while (j < width + 2);

            A += 2 * buf_stride;
            b += 2 * buf_stride;
            C += 2 * buf_stride;
            D += 2 * buf_stride;
            i += 2;
        } while (i < height + 2);
    } else {
        do {
            int32_t j = 0;
            do {
                const __m512i sum1 = boxsum_from_ii_512(D + j, buf_stride, r);
                const __m512i sum2 = boxsum_from_ii_512(C + j, buf_stride, r);
                const __m512i p    = compute_p_highbd_512(sum1, sum2, bit_depth, n);
                const __m512i z    = _mm512_min_epi32(
                    _mm512_srli_epi32(_mm512_add_epi32(_mm512_mullo_epi32(p, s), rnd_z),
                                      SGRPROJ_MTABLE_BITS),
                    _mm512_set1_epi32(255));
                const __m512i a_res = _mm512_i32gather_epi32(z ,eb_x_by_xplus1 , 4);
                yy_storeu_512(A + j, a_res);

                const __m512i a_complement =
                    _mm512_sub_epi32(_mm512_set1_epi32(SGRPROJ_SGR), a_res);

                // sum1 might have lanes greater than 2^15, so we can't use madd to do
                // multiplication involving sum1. However, a_complement and one_over_n
                // are both less than 256, so we can multiply them first.
                const __m512i a_comp_over_n = _mm512_madd_epi16(a_complement, one_over_n);
                const __m512i b_int         = _mm512_mullo_epi32(a_comp_over_n, sum1);
                const __m512i b_res =
                    _mm512_srli_epi32(_mm512_add_epi32(b_int, rnd_res), SGRPROJ_RECIP_BITS);
                yy_storeu_512(b + j, b_res);
                j += 16;
            } while (j < width + 2);

            A += 2 * buf_stride;
            b += 2 * buf_stride;
            C += 2 * buf_stride;
            D += 2 * buf_stride;
            i += 2;
        } while (i < height + 2);
    }
}

// Calculate 8 values of the "cross sum" starting at buf.
//
// Pixels are indexed like this:
// xtl  xt   xtr
//  -   buf   -
// xbl  xb   xbr
//
// Pixels are weighted like this:
//  5    6    5
//  0    0    0
//  5    6    5
//
// fives = xtl + xtr + xbl + xbr
// sixes = xt + xb
// cross_sum = 6 * sixes + 5 * fives
//           = 5 * (fives + sixes) - sixes
//           = (fives + sixes) << 2 + (fives + sixes) + sixes
static INLINE __m256i cross_sum_fast_even_row(const int32_t *buf, int32_t stride) {
    const __m256i xtl = yy_loadu_256(buf - 1 - stride);
    const __m256i xt  = yy_loadu_256(buf - stride);
    const __m256i xtr = yy_loadu_256(buf + 1 - stride);
    const __m256i xbl = yy_loadu_256(buf - 1 + stride);
    const __m256i xb  = yy_loadu_256(buf + stride);
    const __m256i xbr = yy_loadu_256(buf + 1 + stride);

    const __m256i fives = _mm256_add_epi32(xtl, _mm256_add_epi32(xtr, _mm256_add_epi32(xbr, xbl)));
    const __m256i sixes = _mm256_add_epi32(xt, xb);
    const __m256i fives_plus_sixes = _mm256_add_epi32(fives, sixes);

    return _mm256_add_epi32(
        _mm256_add_epi32(_mm256_slli_epi32(fives_plus_sixes, 2), fives_plus_sixes), sixes);
}

// Calculate 8 values of the "cross sum" starting at buf.
//
// Pixels are indexed like this:
// xl    x   xr
//
// Pixels are weighted like this:
//  5    6    5
//
// buf points to x
//
// fives = xl + xr
// sixes = x
// cross_sum = 5 * fives + 6 * sixes
//           = 4 * (fives + sixes) + (fives + sixes) + sixes
//           = (fives + sixes) << 2 + (fives + sixes) + sixes
static INLINE __m256i cross_sum_fast_odd_row(const int32_t *buf) {
    const __m256i xl = yy_loadu_256(buf - 1);
    const __m256i x  = yy_loadu_256(buf);
    const __m256i xr = yy_loadu_256(buf + 1);

    const __m256i fives = _mm256_add_epi32(xl, xr);
    const __m256i sixes = x;

    const __m256i fives_plus_sixes = _mm256_add_epi32(fives, sixes);

    return _mm256_add_epi32(
        _mm256_add_epi32(_mm256_slli_epi32(fives_plus_sixes, 2), fives_plus_sixes), sixes);
}

// The final filter for the self-guided restoration. Computes a
// weighted average across A, b with "cross sums" (see cross_sum_...
// implementations above).
static AOM_FORCE_INLINE void final_filter_fast(int32_t *dst, int32_t dst_stride, const int32_t *A,
                                               const int32_t *B, int32_t buf_stride,
                                               const uint8_t *dgd8, int32_t dgd_stride,
                                               int32_t width, int32_t height, int32_t highbd) {
    const int32_t nb0       = 5;
    const int32_t nb1       = 4;
    const __m256i rounding0 = round_for_shift(SGRPROJ_SGR_BITS + nb0 - SGRPROJ_RST_BITS);
    const __m256i rounding1 = round_for_shift(SGRPROJ_SGR_BITS + nb1 - SGRPROJ_RST_BITS);
    int32_t       i         = 0;

    if (!highbd) {
        do {
            if (!(i & 1)) { // even row
                int32_t j = 0;
                do {
                    const __m256i a   = cross_sum_fast_even_row(A + j, buf_stride);
                    const __m256i b   = cross_sum_fast_even_row(B + j, buf_stride);
                    const __m128i raw = xx_loadl_64(dgd8 + j);
                    const __m256i src = _mm256_cvtepu8_epi32(raw);
                    const __m256i v   = _mm256_add_epi32(_mm256_madd_epi16(a, src), b);
                    const __m256i w   = _mm256_srai_epi32(_mm256_add_epi32(v, rounding0),
                                                        SGRPROJ_SGR_BITS + nb0 - SGRPROJ_RST_BITS);
                    yy_storeu_256(dst + j, w);
                    j += 8;
                } while (j < width);
            } else { // odd row
                int32_t j = 0;
                do {
                    const __m256i a   = cross_sum_fast_odd_row(A + j);
                    const __m256i b   = cross_sum_fast_odd_row(B + j);
                    const __m128i raw = xx_loadl_64(dgd8 + j);
                    const __m256i src = _mm256_cvtepu8_epi32(raw);
                    const __m256i v   = _mm256_add_epi32(_mm256_madd_epi16(a, src), b);
                    const __m256i w   = _mm256_srai_epi32(_mm256_add_epi32(v, rounding1),
                                                        SGRPROJ_SGR_BITS + nb1 - SGRPROJ_RST_BITS);
                    yy_storeu_256(dst + j, w);
                    j += 8;
                } while (j < width);
            }

            A += buf_stride;
            B += buf_stride;
            dgd8 += dgd_stride;
            dst += dst_stride;
        } while (++i < height);
    } else {
        const uint16_t *dgd_real = CONVERT_TO_SHORTPTR(dgd8);

        do {
            if (!(i & 1)) { // even row
                int32_t j = 0;
                do {
                    const __m256i a   = cross_sum_fast_even_row(A + j, buf_stride);
                    const __m256i b   = cross_sum_fast_even_row(B + j, buf_stride);
                    const __m128i raw = xx_loadu_128(dgd_real + j);
                    const __m256i src = _mm256_cvtepu16_epi32(raw);
                    const __m256i v   = _mm256_add_epi32(_mm256_madd_epi16(a, src), b);
                    const __m256i w   = _mm256_srai_epi32(_mm256_add_epi32(v, rounding0),
                                                        SGRPROJ_SGR_BITS + nb0 - SGRPROJ_RST_BITS);
                    yy_storeu_256(dst + j, w);
                    j += 8;
                } while (j < width);
            } else { // odd row
                int32_t j = 0;
                do {
                    const __m256i a   = cross_sum_fast_odd_row(A + j);
                    const __m256i b   = cross_sum_fast_odd_row(B + j);
                    const __m128i raw = xx_loadu_128(dgd_real + j);
                    const __m256i src = _mm256_cvtepu16_epi32(raw);
                    const __m256i v   = _mm256_add_epi32(_mm256_madd_epi16(a, src), b);
                    const __m256i w   = _mm256_srai_epi32(_mm256_add_epi32(v, rounding1),
                                                        SGRPROJ_SGR_BITS + nb1 - SGRPROJ_RST_BITS);
                    yy_storeu_256(dst + j, w);
                    j += 8;
                } while (j < width);
            }

            A += buf_stride;
            B += buf_stride;
            dgd_real += dgd_stride;
            dst += dst_stride;
        } while (++i < height);
    }
}

static INLINE __m512i cross_sum_fast_even_row_512(const int32_t *buf, int32_t stride) {
    const __m512i xtl = yy_loadu_512(buf - 1 - stride);
    const __m512i xt  = yy_loadu_512(buf - stride);
    const __m512i xtr = yy_loadu_512(buf + 1 - stride);
    const __m512i xbl = yy_loadu_512(buf - 1 + stride);
    const __m512i xb  = yy_loadu_512(buf + stride);
    const __m512i xbr = yy_loadu_512(buf + 1 + stride);

    const __m512i fives = _mm512_add_epi32(xtl, _mm512_add_epi32(xtr, _mm512_add_epi32(xbr, xbl)));
    const __m512i sixes = _mm512_add_epi32(xt, xb);
    const __m512i fives_plus_sixes = _mm512_add_epi32(fives, sixes);

    return _mm512_add_epi32(
        _mm512_add_epi32(_mm512_slli_epi32(fives_plus_sixes, 2), fives_plus_sixes), sixes);
}

static INLINE __m512i cross_sum_fast_odd_row_512(const int32_t *buf) {
    const __m512i xl = yy_loadu_512(buf - 1);
    const __m512i x  = yy_loadu_512(buf);
    const __m512i xr = yy_loadu_512(buf + 1);

    const __m512i fives = _mm512_add_epi32(xl, xr);
    const __m512i sixes = x;

    const __m512i fives_plus_sixes = _mm512_add_epi32(fives, sixes);

    return _mm512_add_epi32(
        _mm512_add_epi32(_mm512_slli_epi32(fives_plus_sixes, 2), fives_plus_sixes), sixes);
}


static AOM_FORCE_INLINE void final_filter_fast_512(int32_t *dst, int32_t dst_stride, const int32_t *A,
                                               const int32_t *B, int32_t buf_stride,
                                               const uint8_t *dgd8, int32_t dgd_stride,
                                               int32_t width, int32_t height, int32_t highbd) {
    const int32_t nb0       = 5;
    const int32_t nb1       = 4;
    const __m512i rounding0 = round_for_shift_512(SGRPROJ_SGR_BITS + nb0 - SGRPROJ_RST_BITS);
    const __m512i rounding1 = round_for_shift_512(SGRPROJ_SGR_BITS + nb1 - SGRPROJ_RST_BITS);
    int32_t       i         = 0;

    if (!highbd) {
        do {
            if (!(i & 1)) { // even row
                int32_t j = 0;
                do {
                    const __m512i a   = cross_sum_fast_even_row_512(A + j, buf_stride);
                    const __m512i b   = cross_sum_fast_even_row_512(B + j, buf_stride);
                    const __m128i raw = xx_loadu_128(dgd8 + j);
                    const __m512i src = _mm512_cvtepu8_epi32(raw);
                    const __m512i v   = _mm512_add_epi32(_mm512_madd_epi16(a, src), b);
                    const __m512i w   = _mm512_srai_epi32(_mm512_add_epi32(v, rounding0),
                                                        SGRPROJ_SGR_BITS + nb0 - SGRPROJ_RST_BITS);
                    yy_storeu_512(dst + j, w);
                    j += 16;
                } while (j < width);
            } else { // odd row
                int32_t j = 0;
                do {
                    const __m512i a   = cross_sum_fast_odd_row_512(A + j);
                    const __m512i b   = cross_sum_fast_odd_row_512(B + j);
                    const __m128i raw = xx_loadu_128(dgd8 + j);
                    const __m512i src = _mm512_cvtepu8_epi32(raw);
                    const __m512i v   = _mm512_add_epi32(_mm512_madd_epi16(a, src), b);
                    const __m512i w   = _mm512_srai_epi32(_mm512_add_epi32(v, rounding1),
                                                        SGRPROJ_SGR_BITS + nb1 - SGRPROJ_RST_BITS);
                    yy_storeu_512(dst + j, w);
                    j += 16;
                } while (j < width);
            }

            A += buf_stride;
            B += buf_stride;
            dgd8 += dgd_stride;
            dst += dst_stride;
        } while (++i < height);
    } else {
        const uint16_t *dgd_real = CONVERT_TO_SHORTPTR(dgd8);

        do {
            if (!(i & 1)) { // even row
                int32_t j = 0;
                do {
                    const __m512i a   = cross_sum_fast_even_row_512(A + j, buf_stride);
                    const __m512i b   = cross_sum_fast_even_row_512(B + j, buf_stride);
                    const __m256i raw = xx_loadu_256(dgd_real + j);
                    const __m512i src = _mm512_cvtepu16_epi32(raw);
                    const __m512i v   = _mm512_add_epi32(_mm512_madd_epi16(a, src), b);
                    const __m512i w   = _mm512_srai_epi32(_mm512_add_epi32(v, rounding0),
                                                        SGRPROJ_SGR_BITS + nb0 - SGRPROJ_RST_BITS);
                    yy_storeu_512(dst + j, w);
                    j += 16;
                } while (j < width);
            } else { // odd row
                int32_t j = 0;
                do {
                    const __m512i a   = cross_sum_fast_odd_row_512(A + j);
                    const __m512i b   = cross_sum_fast_odd_row_512(B + j);
                    const __m256i raw = xx_loadu_256(dgd_real + j);
                    const __m512i src = _mm512_cvtepu16_epi32(raw);
                    const __m512i v   = _mm512_add_epi32(_mm512_madd_epi16(a, src), b);
                    const __m512i w   = _mm512_srai_epi32(_mm512_add_epi32(v, rounding1),
                                                        SGRPROJ_SGR_BITS + nb1 - SGRPROJ_RST_BITS);
                    yy_storeu_512(dst + j, w);
                    j += 16;
                } while (j < width);
            }

            A += buf_stride;
            B += buf_stride;
            dgd_real += dgd_stride;
            dst += dst_stride;
        } while (++i < height);
    }
}

void eb_av1_selfguided_restoration_avx2(const uint8_t *dgd8, int32_t width, int32_t height,
                                        int32_t dgd_stride, int32_t *flt0, int32_t *flt1,
                                        int32_t flt_stride, int32_t sgr_params_idx,
                                        int32_t bit_depth, int32_t highbd) {

    //printf("w %d, h %d, dgd_stride %d, flt_stride %d, sgr %d, bit_depth %d, highbd %d, ",
    //width, height, dgd_stride, flt_stride, sgr_params_idx, bit_depth, highbd);

    // The ALIGN_POWER_OF_TWO macro here ensures that column 1 of atl, btl,
    // ctl and dtl is 32-byte aligned.
    const int32_t buf_elts = ALIGN_POWER_OF_TWO(RESTORATION_PROC_UNIT_PELS, 3);
    //printf("bug_elts %d, ",buf_elts);
    DECLARE_ALIGNED(32, int32_t, buf[4 * ALIGN_POWER_OF_TWO(RESTORATION_PROC_UNIT_PELS, 3)]);

    const int32_t width_ext  = width + 2 * SGRPROJ_BORDER_HORZ;
    const int32_t height_ext = height + 2 * SGRPROJ_BORDER_VERT;
    //printf("w_ext %d, h_ext %d, ", width_ext, height_ext);
    // Adjusting the stride of A and b here appears to avoid bad cache effects,
    // leading to a significant speed improvement.
    // We also align the stride to a multiple of 32 bytes for efficiency.
    int32_t buf_stride = ALIGN_POWER_OF_TWO(width_ext + 16, 3);
    //printf("buf_stride %d, ",buf_stride);
    // The "tl" pointers point at the top-left of the initialised data for the
    // array.
    int32_t *atl = buf + 0 * buf_elts + 7;
    int32_t *btl = buf + 1 * buf_elts + 7;
    int32_t *ctl = buf + 2 * buf_elts + 7;
    int32_t *dtl = buf + 3 * buf_elts + 7;

    // The "0" pointers are (- SGRPROJ_BORDER_VERT, -SGRPROJ_BORDER_HORZ). Note
    // there's a zero row and column in A, b (integral images), so we move down
    // and right one for them.
    const int32_t buf_diag_border = SGRPROJ_BORDER_HORZ + buf_stride * SGRPROJ_BORDER_VERT;
    //printf("buf_diag_border %d, ",buf_diag_border);
    int32_t *a0 = atl + 1 + buf_stride;
    int32_t *b0 = btl + 1 + buf_stride;
    int32_t *c0 = ctl + 1 + buf_stride;
    int32_t *d0 = dtl + 1 + buf_stride;

    // Finally, A, b, C, D point at position (0, 0).
    int32_t *A = a0 + buf_diag_border;
    int32_t *b = b0 + buf_diag_border;
    int32_t *C = c0 + buf_diag_border;
    int32_t *D = d0 + buf_diag_border;

    const int32_t  dgd_diag_border = SGRPROJ_BORDER_HORZ + dgd_stride * SGRPROJ_BORDER_VERT;
    const uint8_t *dgd0            = dgd8 - dgd_diag_border;
    //printf("buf_diag_border %d ",buf_diag_border);
    // Generate integral images from the input. C will contain sums of squares; D
    // will contain just sums
/*
    if (highbd)
        integral_images_highbd(
            CONVERT_TO_SHORTPTR(dgd0), dgd_stride, width_ext, height_ext, ctl, dtl, buf_stride);
    else
        integral_images_512(dgd0, dgd_stride, width_ext, height_ext, ctl, dtl, buf_stride);
*/
    if (highbd)
        integral_images_highbd_512(
            CONVERT_TO_SHORTPTR(dgd0), dgd_stride, width_ext, height_ext, ctl, dtl, buf_stride);
    else
        integral_images_512(dgd0, dgd_stride, width_ext, height_ext, ctl, dtl, buf_stride);

    const SgrParamsType *const params = &eb_sgr_params[sgr_params_idx];
    // Write to flt0 and flt1
    // If params->r == 0 we skip the corresponding filter. We only allow one of
    // the radii to be 0, as having both equal to 0 would be equivalent to
    // skipping SGR entirely.
    assert(!(params->r[0] == 0 && params->r[1] == 0));
    assert(params->r[0] < AOMMIN(SGRPROJ_BORDER_VERT, SGRPROJ_BORDER_HORZ));
    assert(params->r[1] < AOMMIN(SGRPROJ_BORDER_VERT, SGRPROJ_BORDER_HORZ));
    //printf("sgr_params_idx %d, ", sgr_params_idx);
    //printf("params_r0 %d, params_r1 %d\n", params->r[0], params->r[1]);
    
/*
    if (params->r[0] > 0) {
        calc_ab_fast(A, b, C, D, width, height, buf_stride, bit_depth, sgr_params_idx, 0);
        final_filter_fast(
            flt0, flt_stride, A, b, buf_stride, dgd8, dgd_stride, width, height, highbd);
    }

    if (params->r[1] > 0) {
        calc_ab(A, b, C, D, width, height, buf_stride, bit_depth, sgr_params_idx, 1);
        final_filter(flt1, flt_stride, A, b, buf_stride, dgd8, dgd_stride, width, height, highbd);
    }
*/
    if (params->r[0] > 0) {
        calc_ab_fast_512(A, b, C, D, width, height, buf_stride, bit_depth, sgr_params_idx, 0);
        final_filter_fast_512(
            flt0, flt_stride, A, b, buf_stride, dgd8, dgd_stride, width, height, highbd);
    }

    if (params->r[1] > 0) {
        calc_ab_512(A, b, C, D, width, height, buf_stride, bit_depth, sgr_params_idx, 1);
        final_filter_512(flt1, flt_stride, A, b, buf_stride, dgd8, dgd_stride, width, height, highbd);
    }

}

void eb_apply_selfguided_restoration_avx2(const uint8_t *dat8, int32_t width, int32_t height,
                                          int32_t stride, int32_t eps, const int32_t *xqd,
                                          uint8_t *dst8, int32_t dst_stride, int32_t *tmpbuf,
                                          int32_t bit_depth, int32_t highbd) {
    int32_t *flt0 = tmpbuf;
    int32_t *flt1 = flt0 + RESTORATION_UNITPELS_MAX;
    assert(width * height <= RESTORATION_UNITPELS_MAX);
    eb_av1_selfguided_restoration_avx2(
        dat8, width, height, stride, flt0, flt1, width, eps, bit_depth, highbd);
    const SgrParamsType *const params = &eb_sgr_params[eps];
    int32_t                    xq[2];
    eb_decode_xq(xqd, xq, params);

    //const __m256i xq0      = _mm256_set1_epi32(xq[0]);
    //const __m256i xq1      = _mm256_set1_epi32(xq[1]);
    const __m512i xq0      = _mm512_set1_epi32(xq[0]);
    const __m512i xq1      = _mm512_set1_epi32(xq[1]);
    //const __m256i rounding = round_for_shift(SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);
    const __m512i rounding = round_for_shift_512(SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);

    int32_t i = height;

    if (!highbd) {
        //const __m256i idx = _mm256_setr_epi32(0, 4, 1, 5, 0, 0, 0, 0);
        const __m512i idx = _mm512_setr_epi32(0, 4, 1, 5, 0, 0, 0, 0, 8, 12, 9, 13, 0, 0, 0, 0);

        do {
            // Calculate output in batches of 16 pixels
            int32_t j = 0;
            do {
                /* const __m128i src  = xx_loadu_128(dat8 + j);
                const __m256i ep_0 = _mm256_cvtepu8_epi32(src);
                const __m256i ep_1 = _mm256_cvtepu8_epi32(_mm_srli_si128(src, 8));
                const __m256i u_0  = _mm256_slli_epi32(ep_0, SGRPROJ_RST_BITS);
                const __m256i u_1  = _mm256_slli_epi32(ep_1, SGRPROJ_RST_BITS);
                __m256i       v_0  = _mm256_slli_epi32(u_0, SGRPROJ_PRJ_BITS);
                __m256i       v_1  = _mm256_slli_epi32(u_1, SGRPROJ_PRJ_BITS);*/

                const __m128i src_0  = xx_loadu_128(dat8 + j);
                const __m128i src_1  = xx_loadu_128(dat8 + j + 16);
                const __m512i ep_0 = _mm512_cvtepu8_epi32(src_0);
                const __m512i ep_1 = _mm512_cvtepu8_epi32(src_1);
                const __m512i u_0  = _mm512_slli_epi32(ep_0, SGRPROJ_RST_BITS);
                const __m512i u_1  = _mm512_slli_epi32(ep_1, SGRPROJ_RST_BITS);
                __m512i       v_0  = _mm512_slli_epi32(u_0, SGRPROJ_PRJ_BITS);
                __m512i       v_1  = _mm512_slli_epi32(u_1, SGRPROJ_PRJ_BITS);

                if (params->r[0] > 0) {
                    /*const __m256i f1_0 = _mm256_sub_epi32(yy_loadu_256(&flt0[j + 0]), u_0);
                    const __m256i f1_1 = _mm256_sub_epi32(yy_loadu_256(&flt0[j + 8]), u_1);
                    v_0                = _mm256_add_epi32(v_0, _mm256_mullo_epi32(xq0, f1_0));
                    v_1                = _mm256_add_epi32(v_1, _mm256_mullo_epi32(xq0, f1_1));*/
                    const __m512i f1_0 = _mm512_sub_epi32(yy_loadu_512(&flt0[j + 0]), u_0);
                    const __m512i f1_1 = _mm512_sub_epi32(yy_loadu_512(&flt0[j + 8]), u_1);
                    v_0                = _mm512_add_epi32(v_0, _mm512_mullo_epi32(xq0, f1_0));
                    v_1                = _mm512_add_epi32(v_1, _mm512_mullo_epi32(xq0, f1_1));
                }

                if (params->r[1] > 0) {
                    /*const __m256i f2_0 = _mm256_sub_epi32(yy_loadu_256(&flt1[j + 0]), u_0);
                    const __m256i f2_1 = _mm256_sub_epi32(yy_loadu_256(&flt1[j + 8]), u_1);
                    v_0                = _mm256_add_epi32(v_0, _mm256_mullo_epi32(xq1, f2_0));
                    v_1                = _mm256_add_epi32(v_1, _mm256_mullo_epi32(xq1, f2_1));*/
                    const __m512i f2_0 = _mm512_sub_epi32(yy_loadu_512(&flt1[j + 0]), u_0);
                    const __m512i f2_1 = _mm512_sub_epi32(yy_loadu_512(&flt1[j + 8]), u_1);
                    v_0                = _mm512_add_epi32(v_0, _mm512_mullo_epi32(xq1, f2_0));
                    v_1                = _mm512_add_epi32(v_1, _mm512_mullo_epi32(xq1, f2_1));
                }

                /*const __m256i w_0 = _mm256_srai_epi32(_mm256_add_epi32(v_0, rounding),
                                                      SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);
                const __m256i w_1 = _mm256_srai_epi32(_mm256_add_epi32(v_1, rounding),
                                                      SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);*/
                const __m512i w_0 = _mm512_srai_epi32(_mm512_add_epi32(v_0, rounding),
                                                      SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);
                const __m512i w_1 = _mm512_srai_epi32(_mm512_add_epi32(v_1, rounding),
                                                      SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);

                // Pack into 8 bits and clamp to [0, 256)
                // Note that each pack messes up the order of the bits,
                // so we use a permute function to correct this
                // 0, 1, 4, 5, 2, 3, 6, 7
                //const __m256i tmp = _mm256_packus_epi32(w_0, w_1);
                const __m512i tmp = _mm512_packus_epi32(w_0, w_1);
                // 0, 1, 4, 5, 2, 3, 6, 7, 0, 1, 4, 5, 2, 3, 6, 7
                //const __m256i tmp2 = _mm256_packus_epi16(tmp, tmp);
                const __m512i tmp2 = _mm512_packus_epi16(tmp, tmp);
                // 0, 1, 2, 3, 4, 5, 6, 7, ...
                //const __m256i tmp3 = _mm256_permutevar8x32_epi32(tmp2, idx);
                //const __m128i res  = _mm256_castsi256_si128(tmp3);
                const __m512i tmp3 = _mm512_permutexvar_epi32(idx, tmp2);
                const __m256i res  = _mm512_castsi512_si256(tmp3);
                //xx_storeu_128(dst8 + j, res);
                xx_storeu_256(dst8 + j, res);
                j += 32;
            } while (j < width);

            dat8 += stride;
            flt0 += width;
            flt1 += width;
            dst8 += dst_stride;
        } while (--i);
    } else {
        //const __m256i   max   = _mm256_set1_epi16((1 << bit_depth) - 1);
        const __m512i   max   = _mm512_set1_epi16((1 << bit_depth) - 1);
        const uint16_t *dat16 = CONVERT_TO_SHORTPTR(dat8);
        uint16_t *      dst16 = CONVERT_TO_SHORTPTR(dst8);

        do {
            // Calculate output in batches of 16 pixels
            int32_t j = 0;
            do {
                /*const __m128i src_0 = xx_loadu_128(dat16 + j + 0);
                const __m128i src_1 = xx_loadu_128(dat16 + j + 8);
                const __m256i ep_0  = _mm256_cvtepu16_epi32(src_0);
                const __m256i ep_1  = _mm256_cvtepu16_epi32(src_1);
                const __m256i u_0   = _mm256_slli_epi32(ep_0, SGRPROJ_RST_BITS);
                const __m256i u_1   = _mm256_slli_epi32(ep_1, SGRPROJ_RST_BITS);
                __m256i       v_0   = _mm256_slli_epi32(u_0, SGRPROJ_PRJ_BITS);
                __m256i       v_1   = _mm256_slli_epi32(u_1, SGRPROJ_PRJ_BITS);*/
                const __m256i src_0 = xx_loadu_256(dat16 + j + 0);
                const __m256i src_1 = xx_loadu_256(dat16 + j + 16);
                const __m512i ep_0  = _mm512_cvtepu16_epi32(src_0);
                const __m512i ep_1  = _mm512_cvtepu16_epi32(src_1);
                const __m512i u_0   = _mm512_slli_epi32(ep_0, SGRPROJ_RST_BITS);
                const __m512i u_1   = _mm512_slli_epi32(ep_1, SGRPROJ_RST_BITS);
                __m512i       v_0   = _mm512_slli_epi32(u_0, SGRPROJ_PRJ_BITS);
                __m512i       v_1   = _mm512_slli_epi32(u_1, SGRPROJ_PRJ_BITS);

                if (params->r[0] > 0) {
                    /*const __m256i f1_0 = _mm256_sub_epi32(yy_loadu_256(&flt0[j + 0]), u_0);
                    const __m256i f1_1 = _mm256_sub_epi32(yy_loadu_256(&flt0[j + 8]), u_1);
                    v_0                = _mm256_add_epi32(v_0, _mm256_mullo_epi32(xq0, f1_0));
                    v_1                = _mm256_add_epi32(v_1, _mm256_mullo_epi32(xq0, f1_1));*/
                    const __m512i f1_0 = _mm512_sub_epi32(yy_loadu_512(&flt0[j + 0]), u_0);
                    const __m512i f1_1 = _mm512_sub_epi32(yy_loadu_512(&flt0[j + 16]), u_1);
                    v_0                = _mm512_add_epi32(v_0, _mm512_mullo_epi32(xq0, f1_0));
                    v_1                = _mm512_add_epi32(v_1, _mm512_mullo_epi32(xq0, f1_1));
                }

                if (params->r[1] > 0) {
                    /*const __m256i f2_0 = _mm256_sub_epi32(yy_loadu_256(&flt1[j + 0]), u_0);
                    const __m256i f2_1 = _mm256_sub_epi32(yy_loadu_256(&flt1[j + 8]), u_1);
                    v_0                = _mm256_add_epi32(v_0, _mm256_mullo_epi32(xq1, f2_0));
                    v_1                = _mm256_add_epi32(v_1, _mm256_mullo_epi32(xq1, f2_1));*/
                    const __m512i f2_0 = _mm512_sub_epi32(yy_loadu_512(&flt1[j + 0]), u_0);
                    const __m512i f2_1 = _mm512_sub_epi32(yy_loadu_512(&flt1[j + 8]), u_1);
                    v_0                = _mm512_add_epi32(v_0, _mm512_mullo_epi32(xq1, f2_0));
                    v_1                = _mm512_add_epi32(v_1, _mm512_mullo_epi32(xq1, f2_1));
                }

                /*const __m256i w_0 = _mm256_srai_epi32(_mm256_add_epi32(v_0, rounding),
                                                      SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);
                const __m512i w_1 = _mm512_srai_epi32(_mm512_add_epi32(v_1, rounding),
                                                      SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);*/

                const __m512i w_0 = _mm512_srai_epi32(_mm512_add_epi32(v_0, rounding),
                                                      SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);
                const __m512i w_1 = _mm512_srai_epi32(_mm512_add_epi32(v_1, rounding),
                                                      SGRPROJ_PRJ_BITS + SGRPROJ_RST_BITS);

                // Pack into 16 bits and clamp to [0, 2^bit_depth)
                // Note that packing into 16 bits messes up the order of the bits,
                // so we use a permute function to correct this
                /*const __m256i tmp  = _mm256_packus_epi32(w_0, w_1);
                const __m256i tmp2 = _mm256_permute4x64_epi64(tmp, 0xd8);
                const __m256i res  = _mm256_min_epi16(tmp2, max);
                yy_storeu_256(dst16 + j, res);*/
                const __m512i tmp  = _mm512_packus_epi32(w_0, w_1);
                const __m512i tmp2 = _mm512_permutexvar_epi64(tmp, _mm512_set_epi64(0, 2, 4, 6, 1, 3, 5, 7));
                const __m512i res  = _mm512_min_epi16(tmp2, max);
                yy_storeu_512(dst16 + j, res);
                j += 16;
            } while (j < width);

            dat16 += stride;
            flt0 += width;
            flt1 += width;
            dst16 += dst_stride;
        } while (--i);
    }
}
