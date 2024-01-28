#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>

void img2col(float* data_img, int channels, int height, int width, int ksize, int stride, int pad, float* data_col) {
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_img = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int row_img = h_offset + h * stride;
                int col_img = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                row_img -= pad;
                col_img -= pad;
                if (row_img >= 0 && col_img >= 0 && row_img < height && col_img < width) {
                    data_col[col_index] = data_img[col_img + width * (row_img + height * c_img)];
                }
            }
        }
    }
}

void matmul(int M, int N, int K, float* A, float* B, float* C) {
    int i, j, k;
    for (i = 0; i < M; ++i) {
        for (k = 0; k < K; ++k) {
            for (j = 0; j < N; ++j) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void transform_g(float* g, float* transformed_g, int in_channels, int out_channels, const int m, const int r) {
    const int in_tile_size = m + r - 1;
    float Gg[12] = {0};
    float G[12] = {1, 0, 0, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0, 0, 1};
    float GT[12] = {1, 0.5, 0.5, 0, 0, 0.5, -0.5, 0, 0, 0.5, 0.5, 1};
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            memset(Gg, 0, in_tile_size * r * sizeof(float));
            matmul(in_tile_size, r, r, G, oc * in_channels * 9 + ic * 9 + g, Gg);
            matmul(in_tile_size, in_tile_size, r, Gg, GT, oc * in_channels * in_tile_size * in_tile_size + ic * in_tile_size * in_tile_size + transformed_g);
        }
    }
}

__global__ void transform_d(float* in, int channels, int height, int width, int stride, int m, int r, float* d, int height_col, int width_col, const int in_tile_size) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    int tile_stride = in_tile_size - (r - stride);  // 相邻 tile 之间首元素的距离

    int tile_size = in_tile_size * in_tile_size;
    int d_height_stride = width_col * tile_size;          // 输入 d 每行的元素数
    int d_channel_stride = height_col * d_height_stride;  // 输入 d 每个通道的元素数
    int in_height_stride = width * tile_stride;
    int in_channel_stride = height * width;

    for (int c = 0; c < channels; ++c) {
        if (w < width_col && h < height_col) {
            for (int i = 0; i < in_tile_size; ++i) {
                for (int j = 0; j < in_tile_size; ++j) {
                    int d_idx = c * d_channel_stride + h * d_height_stride + w * tile_size + i * in_tile_size + j;  // 块内连续
                    int in_idx = c * in_channel_stride + h * in_height_stride + w * tile_stride + i * width + j;    // 块内不连续
                    d[d_idx] = in[in_idx];
                }
            }
        }
    }
}

__device__ void winograd_2d(float* U, float* d, float* result) {
    float BTd[16] = {0};
    float V[16] = {0};
    float UV[16] = {0};
    float ATUV[8] = {0};

    // dot(BT, 4, 4, d, 4, 4, BTd);
    #pragma unroll
    for (int i = 0; i < 4; ++i) BTd[i] = d[0 + i] - d[8 + i];
    #pragma unroll
    for (int i = 0; i < 4; ++i) BTd[4 + i] = d[4 + i] + d[8 + i];
    #pragma unroll
    for (int i = 0; i < 4; ++i) BTd[8 + i] = -d[4 + i] + d[8 + i];
    #pragma unroll
    for (int i = 0; i < 4; ++i) BTd[12 + i] = d[4 + i] - d[12 + i];

    // dot(BTd, 4, 4, B, 4, 4, V);
    #pragma unroll
    for (int i = 0; i < 4; ++i) V[0 + i * 4] = BTd[0 + i * 4] - BTd[2 + i * 4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) V[1 + i * 4] = BTd[1 + i * 4] + BTd[2 + i * 4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) V[2 + i * 4] = -BTd[1 + i * 4] + BTd[2 + i * 4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) V[3 + i * 4] = BTd[1 + i * 4] - BTd[3 + i * 4];

    // UV = U * V
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            UV[4 * i + j] = U[4 * i + j] * V[4 * i + j];
        }
    }

    // dot(AT, 2, 4, UV, 4, 4, ATUV);
    #pragma unroll
    for (int i = 0; i < 4; ++i) ATUV[i] = UV[0 + i] + UV[4 + i] + UV[8 + i];
    #pragma unroll
    for (int i = 0; i < 4; ++i) ATUV[4 + i] = UV[4 + i] - UV[8 + i] - UV[12 + i];

    // dot(ATUV, 2, 4, A, 4, 2, result);
    result[0] += (ATUV[0] + ATUV[1] + ATUV[2]);
    result[2] += (ATUV[4] + ATUV[5] + ATUV[6]);
    result[1] += (ATUV[1] - ATUV[2] - ATUV[3]);
    result[3] += (ATUV[5] - ATUV[6] - ATUV[7]);
}

__global__ void conv_winograd(float* g, float* d, float* o, int height_col, int width_col, int in_channels, int out_channels, const int in_tile_size, const int out_tile_size) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;

    int g_channel_stride = in_channels * in_tile_size * in_tile_size;  // 卷积核 g 每个通道的元素数
    int d_height_stride = width_col * in_tile_size * in_tile_size;     // 输入 d 每行的元素数
    int d_channel_stride = height_col * d_height_stride;               // 输入 d 每个通道的元素数
    int o_height_stride = width_col * out_tile_size * out_tile_size;   // 输出 o 每行的元素数
    int o_channel_stride = height_col * o_height_stride;               // 输出 o 每个通道的元素数

    for (int oc = 0; oc < out_channels; ++oc) {
        for (int ic = 0; ic < in_channels; ++ic) {
            if (w < width_col && h < height_col) {
                winograd_2d(g + oc * g_channel_stride + ic * in_tile_size * in_tile_size,
                            d + ic * d_channel_stride + h * d_height_stride + w * in_tile_size * in_tile_size,
                            o + oc * o_channel_stride + h * o_height_stride + w * out_tile_size * out_tile_size);
            }
        }
    }
}

__global__ void transform_o(
    float* o, int channels, int height, int width, int ksize, int stride, int pad, int m, float* out, int height_col, int width_col, int height_out, int width_out, const int out_tile_size) {
    // o:   height_col x width_col
    // out: (height_col * out_tile_size) x (width_col / out_tile_size)

    int l = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    int channel_stride = width_out * height_out;
    int out_height_stride = width_out * out_tile_size;  // 每次迭代跳过 out_tile_size 行

    int tile_size = out_tile_size * out_tile_size;
    int o_height_stride = width_col * tile_size;

    for (int c = 0; c < channels; ++c) {
        if (k < width_col && l < height_col) {
            for (int i = 0; i < out_tile_size; ++i) {                                                                      // h 方向
                for (int j = 0; j < out_tile_size; ++j) {                                                                  // w 方向
                    int out_idx = c * channel_stride + (l * out_height_stride + i * width_out) + (k * out_tile_size + j);  // 块内不连续
                    int o_idx = c * channel_stride + l * o_height_stride + (k * tile_size + i * out_tile_size + j);        // 块内连续
                    out[out_idx] = o[o_idx];
                }
            }
        }
    }
}

void check_result(float* A, float* B, int len) {
    for (int i = 0; i < len; ++i) {
        if (A[i] != B[i]) {
            printf("error in index\n");
            return;
        }
    }
}

int main() {
    const int m = 2, r = 3, ksize = r, stride = 1, pad = 0;
    const int in_tile_size = m + r - 1;
    const int out_tile_size = m;

    int w = 514, h = 514, c = 8, n = 8;
    int height_out = (h + 2 * pad - ksize) / stride + 1;
    int width_out = (w + 2 * pad - ksize) / stride + 1;
    int height_col = height_out / m;  // 竖直方向上的 tile 数
    int width_col = width_out / m;    // 水平方向上的 tile 数

    float* d = (float*)malloc(w * h * c * sizeof(float));
    float* g = (float*)malloc(ksize * ksize * c * n * sizeof(float));
    for (int i = 0; i < w * h * c; ++i) {
        d[i] = rand() % 10;
    }
    for (int i = 0; i < ksize * ksize * c * n; ++i) {
        g[i] = rand() % 10;
    }

    /* ---- CPU BEGIN ---- */
    float* d_col = (float*)malloc(height_out * width_out * ksize * ksize * c * sizeof(float));
    float* output_cpu = (float*)malloc(width_out * height_out * n * sizeof(float));
    img2col(d, c, h, w, ksize, stride, pad, d_col);
    matmul(n, height_out * width_out, ksize * ksize * c, g, d_col, output_cpu);
    /* ---- CPU END ---- */

    /* ---- GPU BEGIN ---- */
    int d_size = h * w * c * sizeof(float);
    int transformed_d_size = (w - 2) / 2 * (h - 2) / 2 * c * 16 * sizeof(float);
    int transformed_g_size = n * c * 16 * sizeof(float);
    int output_size = width_out * height_out * n * sizeof(float);

    float* transformed_g = (float*)malloc(transformed_g_size);
    float* transformed_o = (float*)malloc(output_size);
    float* output = (float*)malloc(output_size);

    float *transformed_g_cuda, *d_cuda, *transformed_d_cuda, *transformed_o_cuda, *output_cuda;
    cudaMalloc(&d_cuda, d_size);
    cudaMalloc(&transformed_d_cuda, transformed_d_size);
    cudaMalloc(&transformed_g_cuda, transformed_g_size);
    cudaMalloc(&transformed_o_cuda, output_size);
    cudaMalloc(&output_cuda, output_size);

    // transform kernel
    transform_g(g, transformed_g, c, n, m, r);

    // copy kernel to device
    cudaMemcpy(d_cuda, d, d_size, cudaMemcpyHostToDevice);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaMemcpy d failed: %s\n", cudaGetErrorString(err));
    }

    // copy input to device
    cudaMemcpy(transformed_g_cuda, transformed_g, transformed_g_size, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaMemcpy g failed: %s\n", cudaGetErrorString(err));
    }

    const int block_size = 16;  // 每个线程负责一个输入 tile，一个 block 负责 16 * 16 个 tile
    dim3 dim_block(block_size, block_size);
    dim3 dim_grid(width_col / block_size, height_col / block_size);

    // transform input
    transform_d<<<dim_grid, dim_block>>>(d_cuda, c, h, w, stride, m, r, transformed_d_cuda, height_col, width_col, in_tile_size);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("execute kernel transform_d: %s\n", cudaGetErrorString(err));
    }

    // conv
    conv_winograd<<<dim_grid, dim_block>>>(transformed_g_cuda, transformed_d_cuda, transformed_o_cuda, height_col, width_col, c, n, in_tile_size, out_tile_size);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("execute kernel conv_winograd: %s\n", cudaGetErrorString(err));
    }

    // transform back output
    transform_o<<<dim_grid, dim_block>>>(transformed_o_cuda, n, h, w, ksize, stride, pad, m, output_cuda, height_col, width_col, height_out, width_out, out_tile_size);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("execute kernel conv_winograd: %s\n", cudaGetErrorString(err));
    }

    // copy output to host
    cudaMemcpy(output, output_cuda, output_size, cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaMemcpy input failed: %s\n", cudaGetErrorString(err));
    }

    check_result(output, output_cpu, width_out * height_out * n);
    /* ---- GPU END ---- */

    printf("done!\n");
}
