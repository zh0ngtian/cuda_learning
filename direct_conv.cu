#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

void cpu_conv(float *in, float *out, float *cpu_kernel, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int kernel_h, int kernel_w) {
    int out_pos, in_pos, kernel_pos;
    for (int oc = 0; oc < out_c; ++oc) {
        for (int i = 0; i < out_h; ++i) {
            for (int j = 0; j < out_w; ++j) {
                float val = 0;
                out_pos = oc * out_h * out_w + OFFSET(i, j, out_w);
                for (int ic = 0; ic < in_c; ++ic) {
                    for (int ii = 0; ii < kernel_h; ++ii) {
                        for (int jj = 0; jj < kernel_w; ++jj) {
                            in_pos = ic * in_h * in_w + OFFSET(i + ii, j + jj, in_w);
                            kernel_pos = oc * kernel_h * kernel_w + OFFSET(ii, jj, kernel_w);
                            val += in[in_pos] * cpu_kernel[kernel_pos];
                        }
                    }
                }
                out[out_pos] = val;
            }
        }
    }
}

template <const int BLOCK_HEIGHT, const int BLOCK_WIDTH, const int KERNEL_HEIGHT, const int KERNEL_WIDTH, const int MALLOC_TEMP_SIZE, const int MALLOC_BLOCK_HEIGHT, const int MALLOC_BLOCL_WIDTH>
__global__ void gpu_direct_conv(float *in, float *out, float *kernel, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w) {
    int block_y = blockIdx.y;
    int block_x = blockIdx.x;
    int thread_y = threadIdx.y;
    int thread_x = threadIdx.x;
    int thread_num_per_block = BLOCK_HEIGHT * BLOCK_WIDTH;
    int tid = thread_y * BLOCK_WIDTH + thread_x;

    int boundary_y = out_h / BLOCK_HEIGHT - 1;
    int boundary_x = out_w / BLOCK_WIDTH - 1;
    int edge_y = out_h % BLOCK_HEIGHT;
    int edge_x = out_w % BLOCK_WIDTH;

    // global memory --> shared memory
    __shared__ float s_kernel[KERNEL_HEIGHT][KERNEL_WIDTH];
    __shared__ float s_in[MALLOC_BLOCK_HEIGHT][MALLOC_BLOCL_WIDTH];
    float load_reg[4];

    // 当前 block 的起始位置
    int begin_pos = block_y * BLOCK_HEIGHT * in_w + block_x * BLOCK_WIDTH;

    int single_trans_ele_num = 4;                                // 每个线程一次转移 4 个元素，即 tile 大小为 4（边缘 tile 除外）
    int cur_in_block_height = BLOCK_HEIGHT + KERNEL_HEIGHT - 1;  // 当前 block 读入的数据块尺寸
    int cur_in_block_width = BLOCK_WIDTH + KERNEL_WIDTH - 1;     // 当前 block 读入的数据块尺寸
    int in_tile_thread_per_row;                                  // 需要转移的数据块中每行需要的线程数
    int in_tile_row_start;                                       // tile 在 sub-block 中的行坐标
    int in_tile_col;                                             // tile 在 sub-block 中的列坐标
    int in_tile_row_stride;                                      // 同一个线程两次拷贝需要跨越的行数

    // 修正边缘 block 尺寸
    if (block_y == boundary_y) {
        cur_in_block_height = BLOCK_HEIGHT + edge_y + KERNEL_HEIGHT - 1;
    }
    if (block_x == boundary_x) {
        cur_in_block_width = BLOCK_WIDTH + edge_x + KERNEL_WIDTH - 1;
    }

    in_tile_thread_per_row = cur_in_block_width / single_trans_ele_num;
    in_tile_row_start = tid / in_tile_thread_per_row;
    in_tile_col = tid % in_tile_thread_per_row * single_trans_ele_num;
    in_tile_row_stride = thread_num_per_block / in_tile_thread_per_row;

    // copy input block data
    if (in_tile_row_start < cur_in_block_height) {  // 超出当前 block 范围的线程不做拷贝操作
        // 这个循环的迭代次数表示同一个线程会执行几次拷贝，即每个线程在执行完自己那次拷贝以后会跨越 in_tile_row_stride 行再执行一次拷贝
        for (int i = 0; i < cur_in_block_height; i += in_tile_row_stride) {
            FETCH_FLOAT4(load_reg[0]) = FETCH_FLOAT4(in[begin_pos + OFFSET(in_tile_row_start + i, in_tile_col, in_w)]);
            s_in[in_tile_row_start + i][in_tile_col + 0] = load_reg[0];
            s_in[in_tile_row_start + i][in_tile_col + 1] = load_reg[1];
            s_in[in_tile_row_start + i][in_tile_col + 2] = load_reg[2];
            s_in[in_tile_row_start + i][in_tile_col + 3] = load_reg[3];
            // 如果 block 的宽度不能被 tile 宽度整除，对于 block 中的边缘 tile 做特殊处理
            if (in_tile_col + 2 * single_trans_ele_num > cur_in_block_width && in_tile_col + 1 * single_trans_ele_num < cur_in_block_width) {
                for (int j = in_tile_col + 1 * single_trans_ele_num; j < cur_in_block_width; ++j) {
                    s_in[in_tile_row_start + i][j] = in[begin_pos + OFFSET(in_tile_row_start + i, j, in_w)];
                }
            }
        }
    }

    // copy kernel data
    if (thread_x == 0) {
        for (int i = 0; i < KERNEL_HEIGHT / BLOCK_HEIGHT + 1; i++) {
            int yy = i * BLOCK_HEIGHT + thread_y;
            if (yy >= KERNEL_HEIGHT) break;
            for (int j = 0; j < KERNEL_WIDTH; j++) {
                s_kernel[yy][j] = kernel[OFFSET(yy, j, KERNEL_WIDTH)];
            }
        }
    }

    __syncthreads();

    // 以下变量都是针对输出矩阵而言
    int cur_out_block_height = BLOCK_HEIGHT;  // 输出 block 的尺寸
    int cur_out_block_width = BLOCK_WIDTH;    // 输出 block 的尺寸
    int single_calculate_num = 2;             // 每个线程负责计算的输出元素数目
    int out_tile_thread_per_row;              // 输出 block 每行需要的线程数
    int out_tile_row_start;                   // tile 在 sub-block 中的行坐标
    int out_tile_col;                         // tile 在 sub-block 中的列坐标
    int out_tile_row_stride;                  // 同一个线程两次计算需要跨越的行数（同一个 block 中 sub-block 之间相隔的行数）

    // 修正边缘 block 尺寸
    if (block_y == boundary_y) {
        cur_out_block_height = BLOCK_HEIGHT + edge_y;
    }
    if (block_x == boundary_x) {
        cur_out_block_width = BLOCK_WIDTH + edge_x;
    }

    out_tile_thread_per_row = cur_out_block_width / single_calculate_num;
    out_tile_row_start = tid / out_tile_thread_per_row;
    out_tile_col = tid % out_tile_thread_per_row * single_calculate_num;
    out_tile_row_stride = thread_num_per_block / out_tile_thread_per_row;

    float val[MALLOC_TEMP_SIZE];

    for (int oc = 0; oc < out_c; oc++) {
        for (int i = 0; i < MALLOC_TEMP_SIZE; ++i) val[i] = 0;

        for (int ic = 0; ic < in_c; ic++) {
            for (int i = 0; i < cur_out_block_height && (out_tile_row_start + i) < cur_out_block_height; i += out_tile_row_stride) {
                // 如果 block 的宽度不能被 tile 宽度整除，对于 block 中的边缘 tile 做特殊处理
                int new_single_calculate_num = single_calculate_num;
                if (out_tile_col + 2 * single_calculate_num > cur_out_block_width && out_tile_col + 1 * single_calculate_num < cur_out_block_width) {
                    new_single_calculate_num = 3;  // TODO: do not use constant 3
                }

                for (int j = 0; j < new_single_calculate_num; j++) {
                    int temp_pos = i / out_tile_row_stride * new_single_calculate_num + j;
                    for (int ii = 0; ii < KERNEL_HEIGHT; ii++) {
                        for (int jj = 0; jj < KERNEL_WIDTH; jj++) {
                            val[temp_pos] += s_in[out_tile_row_start + i + ii][out_tile_col + j + jj] * s_kernel[ii][jj];
                        }
                    }
                }
            }

            // 预读取下一个 in channel 和对应 kernel in channel
            if (ic < in_c - 1) {
                for (int i = 0; i < cur_in_block_height && in_tile_row_start < cur_in_block_height; i += in_tile_row_stride) {
                    FETCH_FLOAT4(load_reg[0]) = FETCH_FLOAT4(in[begin_pos + (ic + 1) * in_h * in_w + OFFSET(in_tile_row_start + i, in_tile_col, in_w)]);
                    s_in[in_tile_row_start + i][in_tile_col + 0] = load_reg[0];
                    s_in[in_tile_row_start + i][in_tile_col + 1] = load_reg[1];
                    s_in[in_tile_row_start + i][in_tile_col + 2] = load_reg[2];
                    s_in[in_tile_row_start + i][in_tile_col + 3] = load_reg[3];
                    if (in_tile_col + 2 * single_trans_ele_num > cur_in_block_width && cur_in_block_width > in_tile_col + 1 * single_trans_ele_num) {
                        for (int j = in_tile_col + 1 * single_trans_ele_num; j < cur_in_block_width; j++) {
                            s_in[in_tile_row_start + i][j] = in[begin_pos + (ic + 1) * in_h * in_w + OFFSET(in_tile_row_start + i, j, in_w)];
                        }
                    }
                }
                if (thread_x == 0) {
                    for (int i = 0; i < KERNEL_HEIGHT / BLOCK_HEIGHT + 1; i++) {
                        int yy = i * BLOCK_HEIGHT + thread_y;
                        if (yy >= KERNEL_HEIGHT) break;
                        for (int j = 0; j < KERNEL_WIDTH; j++) {
                            s_kernel[yy][j] = kernel[(oc * in_c + ic + 1) * KERNEL_HEIGHT * KERNEL_WIDTH + OFFSET(yy, j, KERNEL_WIDTH)];
                        }
                    }
                }
            }

            __syncthreads();
        }

        // 预读取下一个 kernel out channel
        if (oc + 1 < out_c) {
            if (thread_x == 0) {
                for (int i = 0; i < KERNEL_HEIGHT / BLOCK_HEIGHT + 1; i++) {
                    int yy = i * BLOCK_HEIGHT + thread_y;
                    if (yy >= KERNEL_HEIGHT) break;
                    for (int j = 0; j < KERNEL_WIDTH; j++) {
                        s_kernel[yy][j] = kernel[(oc + 1) * in_c * KERNEL_HEIGHT * KERNEL_WIDTH + OFFSET(yy, j, KERNEL_WIDTH)];
                    }
                }
            }
        }

        __syncthreads();

        // 写回
        int i = 0, j = 0;
        while (i < cur_out_block_height && (out_tile_row_start + i) < cur_out_block_height) {
            // 如果 block 的宽度不能被 tile 宽度整除，对于 block 中的边缘 tile 做特殊处理
            int new_single_calculate_num = single_calculate_num;
            if (out_tile_col + 2 * single_calculate_num > cur_out_block_width && out_tile_col + 1 * single_calculate_num < cur_out_block_width) {
                new_single_calculate_num = 3;  // TODO: do not use constant 3
            }

            while (j < new_single_calculate_num) {
                int out_pos = oc * out_h * out_w + block_y * BLOCK_HEIGHT * out_w + block_x * BLOCK_WIDTH + OFFSET(out_tile_row_start + i, out_tile_col + j, out_w);
                int temp_pos = i / out_tile_row_stride * new_single_calculate_num + j;
                out[out_pos] = val[temp_pos];
                j++;
            }

            i += out_tile_row_stride;
            j = 0;
        }

        // 预读取下一个 in channel
        for (int i = 0; i < cur_in_block_height && in_tile_row_start < cur_in_block_height; i += in_tile_row_stride) {
            FETCH_FLOAT4(load_reg[0]) = FETCH_FLOAT4(in[begin_pos + OFFSET(in_tile_row_start + i, in_tile_col, in_w)]);
            s_in[in_tile_row_start + i][in_tile_col + 0] = load_reg[0];
            s_in[in_tile_row_start + i][in_tile_col + 1] = load_reg[1];
            s_in[in_tile_row_start + i][in_tile_col + 2] = load_reg[2];
            s_in[in_tile_row_start + i][in_tile_col + 3] = load_reg[3];
            if (in_tile_col + 2 * single_trans_ele_num > cur_in_block_width && cur_in_block_width > in_tile_col + 1 * single_trans_ele_num) {
                for (int j = in_tile_col + 1 * single_trans_ele_num; j < cur_in_block_width; j++) {
                    s_in[in_tile_row_start + i][j] = in[begin_pos + OFFSET(in_tile_row_start + i, j, in_w)];
                }
            }
        }
    }
}

int main() {
    const int in_c = 3;
    const int in_h = 1024;
    const int in_w = 1024;
    const int KERNEL_HEIGHT = 6;
    const int KERNEL_WIDTH = 6;
    const int BLOCK_HEIGHT = 8;
    const int BLOCK_WIDTH = 4;
    const int out_c = 3;
    const int out_h = in_h - KERNEL_HEIGHT + 1;
    const int out_w = in_w - KERNEL_WIDTH + 1;

    float *cpu_input, *cpu_output, *cpu_kernel, *cuda_output;
    int input_size = in_c * in_h * in_w;
    int output_size = out_c * out_h * out_w;
    int kernel_size = out_c * in_c * KERNEL_HEIGHT * KERNEL_WIDTH;
    cpu_input = (float *)malloc(input_size * sizeof(float));
    cpu_output = (float *)malloc(output_size * sizeof(float));
    cpu_kernel = (float *)malloc(kernel_size * sizeof(float));
    cuda_output = (float *)malloc(output_size * sizeof(float));
    for (int i = 0; i < input_size; i++) {
        cpu_input[i] = i;
    }
    for (int i = 0; i < kernel_size; ++i) {
        cpu_kernel[i] = 1;
    }
    for (int i = 0; i < output_size; ++i) {
        cpu_output[i] = 0;
        cuda_output[i] = 0;
    }

    /* ---- CPU BEGIN ---- */
    cpu_conv(cpu_input, cpu_output, cpu_kernel, in_c, in_h, in_w, out_c, out_h, out_w, KERNEL_HEIGHT, KERNEL_WIDTH);
    /* ---- CPU END ---- */

    /* ---- GPU BEGIN ---- */
    cudaError_t err;

    float *gpu_input, *gpu_kernel, *gpu_output;
    cudaMalloc(&gpu_input, input_size * sizeof(float));
    cudaMalloc(&gpu_output, output_size * sizeof(float));
    cudaMalloc(&gpu_kernel, kernel_size * sizeof(float));

    cudaMemcpy(gpu_input, cpu_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaMemcpy input failed: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(gpu_kernel, cpu_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaMemcpy kernel failed: %s\n", cudaGetErrorString(err));
    }

    const int MALLOC_BLOCK_HEIGHT = (BLOCK_HEIGHT + KERNEL_HEIGHT) * 2;  // double buffer
    const int MALLOC_BLOCK_WIDTH = (BLOCK_WIDTH + KERNEL_WIDTH) * 2;     // double buffer
    const int MALLOC_TEMP_SIZE = out_c * 8;                              // temp out channel data, TODO

    printf("KERNEL_HEIGHT: %d, KERNEL_WIDTH: %d, MALLOC_BLOCK_HEIGHT: %d, MALLOC_BLOCK_WIDTH: %d, MALLOC_TEMP_SIZE: %d\n",
           KERNEL_HEIGHT,
           KERNEL_WIDTH,
           MALLOC_BLOCK_HEIGHT,
           MALLOC_BLOCK_WIDTH,
           MALLOC_TEMP_SIZE);

    dim3 dim_grid(out_w / BLOCK_WIDTH, out_h / BLOCK_HEIGHT);
    dim3 dim_block(BLOCK_WIDTH, BLOCK_HEIGHT);

    gpu_direct_conv<BLOCK_HEIGHT, BLOCK_WIDTH, KERNEL_HEIGHT, KERNEL_WIDTH, MALLOC_TEMP_SIZE, MALLOC_BLOCK_HEIGHT, MALLOC_BLOCK_WIDTH>
        <<<dim_grid, dim_block>>>(gpu_input, gpu_output, gpu_kernel, in_c, in_h, in_w, out_c, out_h, out_w);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("execute kernel function: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(cuda_output, gpu_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaMemcpy output failed: %s\n", cudaGetErrorString(err));
    }

    for (int i = 0; i < output_size; i++) {
        if (cpu_output[i] != cuda_output[i]) {
            printf("WRONG VALUE: %.2f | %.2f at %d\n", cpu_output[i], cuda_output[i], i);
            break;
        }
    }
    /* ---- GPU END ---- */
}
