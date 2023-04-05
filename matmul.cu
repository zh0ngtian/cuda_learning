#include <vector>
#include <functional>

#include <cublas_v2.h>

#define FETCH_FLOAT4(p) (reinterpret_cast<float4*>(&(p))[0])

#define checkCudaErrors(func) {                                                                  \
    cudaError_t e = (func);                                                                      \
    if (e != cudaSuccess) printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  }

static const char* _cuBlasGetErrorEnum(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "<unknown>";
}

#define checkCuBlasErrors(func) {                                                                           \
    cublasStatus_t e = (func);                                                                              \
    if (e != CUBLAS_STATUS_SUCCESS) printf("%s %d CuBlas: %s", __FILE__, __LINE__, _cuBlasGetErrorEnum(e)); \
  }

const int BLOCK_SIZE_M = 128;
const int BLOCK_SIZE_N = 128;
const int BLOCK_SIZE_K = 8;
const int THREAD_SIZE_Y = 8;
const int THREAD_SIZE_X = 8;
const int THREAD_X_PER_BLOCK = BLOCK_SIZE_N / THREAD_SIZE_X;
const int THREAD_Y_PER_BLOCK = BLOCK_SIZE_M / THREAD_SIZE_Y;
const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;
const int SIZE = THREAD_X_PER_BLOCK;

cublasHandle_t blas_handle;

void setValue(float *data, int num) {
  for (int i = 0; i < num; ++i) {
    data[i] = float(i % 100);
  }
}

void matrixMultiplyCpu(float *A, float *B, float *C, int m, int n, int k) {
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float sum = 0.f;
      for (int kk = 0; kk < k; ++kk) {
        sum += A[mm * k + kk] * B[kk * n + nn];
      }
      C[mm * n + nn] = sum;
    }
  }
}

bool checkEqual(float* v1, float* v2, int num) {
  bool equal = true;
  for (int i = 0; i < num; ++i) {
    float diff = v1[i] - v2[i];
    if (diff > 1e-5 || diff < -1e-5) {
      equal = false;
    }
  }
  if (!equal) {
    for (int i = 0; i < 20; ++i) {
      printf("%f %f\n", v1[i], v2[i]);
    }
  }
  return equal;
}

// baseline: 4.05 ms
__global__ void matrixMultiplyKernel0(float *A, float *B, float *C, int m, int n, int k) {
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  float sum = 0.f;
  if (idx_x < n && idx_y < m) {
    for (int kk = 0; kk < k; ++kk) {
      sum += A[idx_y * k + kk] * B[kk * n + idx_x];
    }
    C[idx_y * n + idx_x] = sum;
  }
}

// use shared memory: 2.40 ms
__global__ void matrixMultiplyKernel1(float *A, float *B, float *C, int m, int n, int k) {
  __shared__ float s_a[SIZE][SIZE];
  __shared__ float s_b[SIZE][SIZE];

  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  float sum = 0.0;
  for (int bk = 0; bk < k; bk += SIZE) {
    s_a[threadIdx.y][threadIdx.x] = A[idx_y * k + bk + threadIdx.x];
    s_b[threadIdx.y][threadIdx.x] = B[(bk + threadIdx.y) * n + idx_x];
    __syncthreads();

    for (int i = 0; i < SIZE; ++i) {
      sum += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
    }
    __syncthreads();
  }

  if (idx_x < n && idx_y < m) {
    C[idx_y * n + idx_x] = sum;
  }
}

// multiply data per thread:
//  - without unroll: 0.82 ms
//  - with unroll: 0.65 ms
__global__ void matrixMultiplyKernel2(float *A, float *B, float *C, int m, int n, int k) {
  __shared__ float s_a[BLOCK_SIZE_M][BLOCK_SIZE_K];
  __shared__ float s_b[BLOCK_SIZE_K][BLOCK_SIZE_N];
  float r_c[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};

  const int tid = threadIdx.y * THREAD_X_PER_BLOCK + threadIdx.x;

  // 每个线程一次只搬运一个数据

  // 在 s_a/s_b 中，当前线程需要搬运的第一个数据的横纵坐标
  const int A_TILE_ROW = tid / BLOCK_SIZE_K;
  const int A_TILE_COL = tid % BLOCK_SIZE_K;
  const int B_TILE_ROW = tid / BLOCK_SIZE_N;
  const int B_TILE_COL = tid % BLOCK_SIZE_N;

  // 在进行多次搬运时需要跨越的行
  const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
  const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;

  for (int bk = 0; bk < k; bk += BLOCK_SIZE_K) {
    // load A from global memory to shared memory
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
      const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW;
      const int col = bk + A_TILE_COL;
      if (blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1) {
        s_a[i + A_TILE_ROW][A_TILE_COL] = row < m && col < k ? A[row * k + col] : 0;
      } else {
        s_a[i + A_TILE_ROW][A_TILE_COL] = A[row * k + col];
      }
    }

    // load B from global memory to shared memory
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
      const int row = bk + i + B_TILE_ROW;
      const int col = BLOCK_SIZE_N * blockIdx.x + B_TILE_COL;
      if (blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1) {
        s_b[i + B_TILE_ROW][B_TILE_COL] = row < k && col < n ? B[row * n + col] : 0;
      } else {
        s_b[i + B_TILE_ROW][B_TILE_COL] = B[row * n + col];
      }
    }

    __syncthreads();

    // 每个线程负责搬运的数据和接下来要计算的数据没有必然联系

    // calculate C
    #pragma unroll
    for (int kk = 0; kk < BLOCK_SIZE_K; ++kk) {
      #pragma unroll
      for (int ty = 0; ty < THREAD_SIZE_Y; ++ty) {
        #pragma unroll
        for (int tx = 0; tx < THREAD_SIZE_X; ++tx) {
            r_c[ty][tx] += s_a[THREAD_SIZE_Y * threadIdx.y + ty][kk] * s_b[kk][THREAD_SIZE_X * threadIdx.x + tx];
        }
      }
    }

    __syncthreads();
  }

  // store back to C
  #pragma unroll
  for (int ty = 0; ty < THREAD_SIZE_Y; ++ty) {
    #pragma unroll
    for (int tx = 0; tx < THREAD_SIZE_X; ++tx) {
      const int row = BLOCK_SIZE_M * blockIdx.y + THREAD_SIZE_Y * threadIdx.y + ty;
      const int col = BLOCK_SIZE_N * blockIdx.x + THREAD_SIZE_X * threadIdx.x + tx;
      if (blockIdx.x == gridDim.x - 1 || blockIdx.y == gridDim.y - 1) {
        if (row < m && col < n) {
          C[row * n + col] += r_c[ty][tx];
        }
      } else {
        C[row * n + col] += r_c[ty][tx];
      }
    }
  }
}

// float4: 0.57 ms
__global__ void matrixMultiplyKernel3(float *A, float *B, float *C, int m, int n, int k) {
  __shared__ float s_a[BLOCK_SIZE_M][BLOCK_SIZE_K];
  __shared__ float s_b[BLOCK_SIZE_K][BLOCK_SIZE_N];
  float r_c[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
  float frag_a[THREAD_SIZE_Y];
  float frag_b[THREAD_SIZE_X];

  const int tid = threadIdx.y * THREAD_X_PER_BLOCK + threadIdx.x;

  // 每个线程一次搬运四个数据

  // 在 s_a/s_b 中，当前线程搬运一行数据需要的线程数
  const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
  const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

  // 在 s_a/s_b 中，当前线程需要搬运的第一个数据组中第一个数据（即四个数据的第一个）的的横纵坐标
  const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
  const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
  const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
  const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

  // 在进行多次搬运时需要跨越的行
  const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
  const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

  for (int bk = 0; bk < k; bk += BLOCK_SIZE_K) {
    // load A from global memory to shared memory
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
      const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW_START;
      const int col = bk + A_TILE_COL;
      FETCH_FLOAT4(s_a[i + A_TILE_ROW_START][A_TILE_COL]) = FETCH_FLOAT4(A[row * k + col]);
    }

    // load B from global memory to shared memory
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
      const int row = bk + i + B_TILE_ROW_START;
      const int col = BLOCK_SIZE_N * blockIdx.x + B_TILE_COL;
      FETCH_FLOAT4(s_b[i + B_TILE_ROW_START][B_TILE_COL]) = FETCH_FLOAT4(B[row * n + col]);
    }

    __syncthreads();

    // 每个线程负责搬运的数据和接下来要计算的数据没有必然联系

    // calculate C
    #pragma unroll
    for (int kk = 0; kk < BLOCK_SIZE_K; ++kk) {
      // load A from shared memory to register
      #pragma unroll
      for (int ty = 0; ty < THREAD_SIZE_Y; ++ty) {
        frag_a[ty] = s_a[THREAD_SIZE_Y * threadIdx.y + ty][kk];
      }

      // load B from shared memory to register
      #pragma unroll
      for (int tx = 0; tx < THREAD_SIZE_X; tx += 4) {
        FETCH_FLOAT4(frag_b[tx]) = FETCH_FLOAT4(s_b[kk][THREAD_SIZE_X * threadIdx.x + tx]);
      }

      #pragma unroll
      for (int ty = 0; ty < THREAD_SIZE_Y; ++ty) {
        #pragma unroll
        for (int tx = 0; tx < THREAD_SIZE_X; ++tx) {
            r_c[ty][tx] += frag_a[ty] * frag_b[tx];
        }
      }
    }
  }

  // store back to C
  #pragma unroll
  for (int ty = 0; ty < THREAD_SIZE_Y; ++ty) {
    #pragma unroll
    for (int tx = 0; tx < THREAD_SIZE_X; tx += 4) {
      const int row = BLOCK_SIZE_M * blockIdx.y + THREAD_SIZE_Y * threadIdx.y + ty;
      const int col = BLOCK_SIZE_N * blockIdx.x + THREAD_SIZE_X * threadIdx.x + tx;
      FETCH_FLOAT4(C[row * n + col]) = FETCH_FLOAT4(r_c[ty][tx]);
    }
  }
}

// double buffer: 0.54 ms
__global__ void matrixMultiplyKernel4(float * A, float * B, float * C, int m, int n, int k) {
  __shared__ float s_a[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
  __shared__ float s_b[2][BLOCK_SIZE_K][BLOCK_SIZE_N];
  float r_c[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
  float frag_a[2][THREAD_SIZE_Y];
  float frag_b[2][THREAD_SIZE_X];

  // 为了存储 BLOCK_SIZE_M * BLOCK_SIZE_K 的数据块，每个线程需要额外开启 ldg_a_reg 个寄存器进行存储
  float ldg_a_reg[BLOCK_SIZE_M * BLOCK_SIZE_K / THREAD_NUM_PER_BLOCK];
  float ldg_b_reg[BLOCK_SIZE_K * BLOCK_SIZE_N / THREAD_NUM_PER_BLOCK];

  const int tid = threadIdx.y * THREAD_X_PER_BLOCK + threadIdx.x;

  // 每个线程一次搬运四个数据

  // 在 s_a/s_b 中，当前线程搬运一行数据需要的线程数
  const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
  const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

  // 在 s_a/s_b 中，当前线程需要搬运的第一个数据组中第一个数据（即四个数据的第一个）的的横纵坐标
  const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
  const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;
  const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
  const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

  // 在进行多次搬运时需要跨越的行
  const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
  const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

  // preload A from global memory to shared memory
  #pragma unroll
  for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
    int ldg_index = i / A_TILE_ROW_STRIDE * 4;
    const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW_START;
    const int col = A_TILE_COL;
    FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[row * k + col]);
    s_a[0][A_TILE_COL + 0][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index + 0];
    s_a[0][A_TILE_COL + 1][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index + 1];
    s_a[0][A_TILE_COL + 2][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index + 2];
    s_a[0][A_TILE_COL + 3][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index + 3];
  }

  // preload B from global memory to shared memory
  #pragma unroll
  for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
    const int row = i + B_TILE_ROW_START;
    const int col = BLOCK_SIZE_N * blockIdx.x + B_TILE_COL;
    FETCH_FLOAT4(s_b[0][i + B_TILE_ROW_START][B_TILE_COL]) = FETCH_FLOAT4(B[row * n + col]);
  }

  __syncthreads();

  // preload A from shared memory to register
  #pragma unroll
  for (int ty = 0; ty < THREAD_SIZE_Y; ty += 4) {
    FETCH_FLOAT4(frag_a[0][ty]) = FETCH_FLOAT4(s_a[0][0][THREAD_SIZE_Y * threadIdx.y + ty]);
  }

  // preload B from shared memory to register
  #pragma unroll
  for (int tx = 0; tx < THREAD_SIZE_X; tx += 4) {
    FETCH_FLOAT4(frag_b[0][tx]) = FETCH_FLOAT4(s_b[0][0][THREAD_SIZE_X * threadIdx.x + tx]);
  }

  int write_stage_idx = 1;
  int bk = 0;
  do {
    bk += BLOCK_SIZE_K;

    if (bk < k) {
      // preload A from global memory to register
      #pragma unroll
      for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW_START;
        const int col = bk + A_TILE_COL;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[row * k + col]);
      }

      // preload B from global memory to register
      #pragma unroll
      for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        int ldg_index = i / B_TILE_ROW_STRIDE * 4;
        const int row = bk + i + B_TILE_ROW_START;
        const int col = BLOCK_SIZE_N * blockIdx.x + B_TILE_COL;
        FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[row * n + col]);
      }
    }

    // 每个线程负责搬运的数据和接下来要计算的数据没有必然联系

    int load_stage_idx = write_stage_idx ^ 1;

    // calculate C
    #pragma unroll
    for (int kk = 0; kk < BLOCK_SIZE_K - 1; ++kk) {
      // preload A from shared memory to register
      #pragma unroll
      for (int ty = 0; ty < THREAD_SIZE_Y; ty += 4) {
        FETCH_FLOAT4(frag_a[(kk + 1) % 2][ty]) = FETCH_FLOAT4(s_a[load_stage_idx][kk + 1][THREAD_SIZE_Y * threadIdx.y + ty]);
      }

      // preload B from shared memory to register
      #pragma unroll
      for (int tx = 0; tx < THREAD_SIZE_X; tx += 4) {
        FETCH_FLOAT4(frag_b[(kk + 1) % 2][tx]) = FETCH_FLOAT4(s_b[load_stage_idx][kk + 1][THREAD_SIZE_X * threadIdx.x + tx]);
      }

      // calculate C (this tile)
      #pragma unroll
      for (int ty = 0; ty < THREAD_SIZE_Y; ++ty) {
        #pragma unroll
        for (int tx = 0; tx < THREAD_SIZE_X; ++tx) {
            r_c[ty][tx] += frag_a[kk % 2][ty] * frag_b[kk % 2][tx];
        }
      }
    }

    if (bk < k) {
      // preload A from register to shared memory
      #pragma unroll
      for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        s_a[write_stage_idx][A_TILE_COL + 0][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index + 0];
        s_a[write_stage_idx][A_TILE_COL + 1][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index + 1];
        s_a[write_stage_idx][A_TILE_COL + 2][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index + 2];
        s_a[write_stage_idx][A_TILE_COL + 3][i + A_TILE_ROW_START] = ldg_a_reg[ldg_index + 3];
      }

      // preload B from register to shared memory
      #pragma unroll
      for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
        int ldg_index = i / B_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(s_b[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
      }

      __syncthreads();
      write_stage_idx ^= 1;
    }

    // preload A from shared memory to register
    #pragma unroll
    for (int ty = 0; ty < THREAD_SIZE_Y; ty += 4) {
      FETCH_FLOAT4(frag_a[0][ty]) = FETCH_FLOAT4(s_a[load_stage_idx ^ 1][0][THREAD_SIZE_Y * threadIdx.y + ty]);
    }

    // preload B from shared memory to register
    #pragma unroll
    for (int tx = 0; tx < THREAD_SIZE_X; tx += 4) {
      FETCH_FLOAT4(frag_b[0][tx]) = FETCH_FLOAT4(s_b[load_stage_idx ^ 1][0][THREAD_SIZE_X * threadIdx.x + tx]);
    }

    // compute last tile matmul THREAD_SIZE_X * THREAD_SIZE_Y
    #pragma unroll
    for (int ty = 0; ty < THREAD_SIZE_Y; ++ty) {
      #pragma unroll
      for (int tx = 0; tx < THREAD_SIZE_X; ++tx) {
          r_c[ty][tx] += frag_a[1][ty] * frag_b[1][tx];
      }
    }

  } while(bk < k);

  // store back to C
  #pragma unroll
  for (int ty = 0; ty < THREAD_SIZE_Y; ++ty) {
    #pragma unroll
    for (int tx = 0; tx < THREAD_SIZE_X; tx += 4) {
      const int row = BLOCK_SIZE_M * blockIdx.y + THREAD_SIZE_Y * threadIdx.y + ty;
      const int col = BLOCK_SIZE_N * blockIdx.x + THREAD_SIZE_X * threadIdx.x + tx;
      FETCH_FLOAT4(C[row * n + col]) = FETCH_FLOAT4(r_c[ty][tx]);
    }
  }
}

void matrixMultiplyImpl0(float * A, float * B, float * C, int m, int n, int k) {
  dim3 block(SIZE, SIZE);
  dim3 grid(n / SIZE, m / SIZE);
  matrixMultiplyKernel0<<<grid, block>>>(A, B, C, m, n, k);
  cudaDeviceSynchronize();
}

void matrixMultiplyImpl1(float * A, float * B, float * C, int m, int n, int k) {
  dim3 block(SIZE, SIZE);
  dim3 grid(n / SIZE, m / SIZE);
  matrixMultiplyKernel1<<<grid, block>>>(A, B, C, m, n, k);
  cudaDeviceSynchronize();
}

void matrixMultiplyImpl2(float * A, float * B, float * C, int m, int n, int k) {
  dim3 block(THREAD_X_PER_BLOCK, THREAD_Y_PER_BLOCK);
  dim3 grid(n / BLOCK_SIZE_N, m / BLOCK_SIZE_M);
  matrixMultiplyKernel2<<<grid, block>>>(A, B, C, m, n, k);
  cudaDeviceSynchronize();
}

void matrixMultiplyImpl3(float * A, float * B, float * C, int m, int n, int k) {
  dim3 block(THREAD_X_PER_BLOCK, THREAD_Y_PER_BLOCK);
  dim3 grid(n / BLOCK_SIZE_N, m / BLOCK_SIZE_M);
  matrixMultiplyKernel3<<<grid, block>>>(A, B, C, m, n, k);
  cudaDeviceSynchronize();
}

void matrixMultiplyImpl4(float * A, float * B, float * C, int m, int n, int k) {
  dim3 block(THREAD_X_PER_BLOCK, THREAD_Y_PER_BLOCK);
  dim3 grid(n / BLOCK_SIZE_N, m / BLOCK_SIZE_M);
  matrixMultiplyKernel4<<<grid, block>>>(A, B, C, m, n, k);
  cudaDeviceSynchronize();
}

// cublas: 0.45 ms
void matrixMultiplyImpl5(float * A, float * B, float * C, int m, int n, int k) {
  float alpha = 1.0;
  float beta = 0.0;
  checkCuBlasErrors(cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n));
}

float matrixMultiplyGpu(float *A, float *B, float *C, int m, int n, int k, int func_idx, int loop_times = 1) {
  std::vector<std::function<void(float*, float*, float*, int, int, int)>> funcs = {
    matrixMultiplyImpl0,
    matrixMultiplyImpl1,
    matrixMultiplyImpl2,
    matrixMultiplyImpl3,
    matrixMultiplyImpl4,
    matrixMultiplyImpl5,
  };

  checkCuBlasErrors(cublasCreate(&blas_handle));

  cudaEvent_t start_time, end_time;
  cudaEventCreate(&start_time);
  cudaEventCreate(&end_time);
  cudaEventRecord(start_time, 0);
  for (int i = 0; i < loop_times; ++i) {
    funcs[func_idx](A, B, C, m, n, k);
  }
  cudaEventRecord(end_time, 0);
  cudaEventSynchronize(start_time);
  cudaEventSynchronize(end_time);

  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    printf("execute kernel function : %s\n", cudaGetErrorString(e));
  }

  float time_cost = 0.0;
  if (loop_times > 1) {
    cudaEventElapsedTime(&time_cost, start_time, end_time);
  }

  return time_cost / loop_times;
}

void run(int nsize) {
  printf("m = n = k = %d\n", nsize);

  int m, n, k;
  m = n = k = nsize;

  int size_a = m * k * sizeof(float);
  int size_b = k * n * sizeof(float);
  int size_c = m * n * sizeof(float);

  float *h_A = (float*)malloc(size_a);
  float *h_B = (float*)malloc(size_b);
  float *h_C = (float*)malloc(size_c);

  setValue(h_A, m * k);
  setValue(h_B, k * n);

  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, size_a);
  cudaMalloc((void **)&d_B, size_b);
  cudaMalloc((void **)&d_C, size_c);
  cudaMemcpy(d_A, h_A, size_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_b, cudaMemcpyHostToDevice);

  bool check_value = false;
  bool benchmark = true;

  if (check_value) {
    int func_idx = 4;

    matrixMultiplyGpu(d_A, d_B, d_C, m, n, k, func_idx);
    cudaMemcpy(h_C, d_C, size_c, cudaMemcpyDeviceToHost);

    float *gt_h_C = (float*)malloc(size_c);
    matrixMultiplyCpu(h_A, h_B, gt_h_C, m, n, k);

    bool is_equal = checkEqual(h_C, gt_h_C, m * n);
    if (!is_equal) {
      printf("=== not equal ===\n");
    } else {
      printf("=== is equal ===\n");
    }
  }

  if (benchmark) {
    for (int func_idx = 0; func_idx < 6; ++func_idx) {
      float time_cost = matrixMultiplyGpu(d_A, d_B, d_C, m, n, k, func_idx, 1000);
      printf("func %d average time cost: %f ms\n", func_idx, time_cost);
    }
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);
}

int main() {
  std::vector<int> nsizes = {128, 256, 512, 1024, 2048};
  for (auto nsize : nsizes) {
    run(nsize);
  }

  return 0;
}
