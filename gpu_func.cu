#include "gpu_func.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>
#include "cublas_v2.h"

#define THREADS_X 2
#define THREADS_Y 512
#define BLOCK_SIZE 32

__global__
void device_add_one(int* d_result, int t) {
    *d_result = t + 1;
}

/*
Just a dummy function that can be used to warm up GPU
*/
int useless_gpu_add_one(int t) {
    int result;
    int* d_result;

    checkCudaErrors(cudaMalloc((void**)&d_result, 1 * sizeof(int)));

    event_pair timer;
    start_timer(&timer);
    device_add_one<<<1,1>>>(d_result, t);
    check_launch("device_add_one");
    double time = stop_timer(&timer);

    std::cout << "device_add_one took: " << time << " seconds" << std::endl;

    checkCudaErrors(cudaMemcpy(&result, d_result, 1 * sizeof(int),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_result));
    return result;
}

/* 
GPU kernel for GEMM using shared memory.
*/
__global__
void gemm_shared_kernel(double* A, double* B, double* C, double alpha, double beta, 
                        int M, int N, int K) {
    int col_local = threadIdx.y;
    int row_local = threadIdx.x;
    int col_global = blockIdx.y * blockDim.y + threadIdx.y;
    int row_global = blockIdx.x * blockDim.x + threadIdx.x; 
    int A_col_global;
    int A_idx_global;
    int B_row_global;
    int B_idx_global;
    int elems;

    // Shared memory for A and B submatrices
    __shared__ double A_sub[BLOCK_SIZE][BLOCK_SIZE+1]; 
    __shared__ double B_sub[BLOCK_SIZE][BLOCK_SIZE+1];

    double dotAB = 0;
    for (int p = 0; p < ((K + BLOCK_SIZE - 1) / BLOCK_SIZE); p++) {
        // Get A_sub from global A
        A_col_global = p * BLOCK_SIZE + col_local;
        if (row_global < M && A_col_global < K) {
            A_idx_global = A_col_global * M + row_global;
            A_sub[row_local][col_local] = A[A_idx_global];
        }

        // Get B_sub from global B
        B_row_global = p * BLOCK_SIZE + row_local;
        if (col_global < N && B_row_global < K) {
            B_idx_global = col_global * K + B_row_global; 
            B_sub[row_local][col_local] = B[B_idx_global];
        }

        // Make sure submatrices are loaded before proceeding
        __syncthreads();

        // Calculate each entry of A_sub * B_sub
        elems = min(BLOCK_SIZE, K - p * BLOCK_SIZE);
        if (row_global < M && col_global < N) {
            for (int e = 0; e < elems; e++) {
                dotAB += A_sub[row_local][e] * B_sub[e][col_local];
            }
        }

        __syncthreads();
    }

    // Multiply by scalars 
    if (row_global < M && col_global < N) {
        C[col_global * M + row_global] = alpha * dotAB + beta * C[col_global * M + row_global];
    }
}

/*
Routine to perform in-place GEMM using shared memory, i.e., C := alpha*A*B + beta*C
*/
int myGEMM(double* __restrict__ A, double* __restrict__ B,
           double* __restrict__ C, double* alpha, double* beta,
           int M, int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */
    int blocks_per_grid_x = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_per_grid_y = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y);

    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    gemm_shared_kernel<<<num_blocks, block_size>>>(A, B, C, *alpha, *beta, M, N, K);

    return 0;
}

/* 
GPU kernel for GEMM. Each thread calculates one element of C using global memory. 
*/
__global__
void gemm_global_kernel(double* A, double* B, double* C, double alpha, double beta, 
                        int M, int N, int K) {
    double dotAB = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        for (int e = 0; e < K; e++) {
            dotAB += A[e * M + row] * B[col * K + e];
        }
        C[col * M + row] = alpha * dotAB + beta * C[col * M + row];
    } 
}

/*
Routine to perform in-place GEMM using global memory, i.e., C := alpha*A*B + beta*C
*/
int myGEMMGlobal(double* __restrict__ A, double* __restrict__ B,
                 double* __restrict__ C, double* alpha, double* beta,
                 int M, int N, int K) {
    /* TODO: Write an efficient GEMM implementation on GPU */
    int threads_per_block_x = THREADS_X;
    int threads_per_block_y = THREADS_Y;
    int blocks_per_grid_x = (N + threads_per_block_x - 1) / threads_per_block_x;
    int blocks_per_grid_y = (M + threads_per_block_y - 1) / threads_per_block_y;

    dim3 block_size(threads_per_block_x, threads_per_block_y);
    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y);

    gemm_global_kernel<<<num_blocks, block_size>>>(A, B, C, *alpha, *beta, M, N, K);

    return 0;
}

/* 
GPU kernel for sigmoid. Each thread calculates one element.
*/
__global__
void sigmoid_kernel(double* A, double* B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = col * M + row;

    if (row < M && col < N) {
        B[idx] = 1 / (1 + exp(-A[idx]));
    }
}

/* 
Routine to perform element-wise sigmoid.
*/
int mySigmoid(double* A, double *B, int M, int N) {
    int threads_per_block_x = THREADS_X;
    int threads_per_block_y = THREADS_Y;
    int blocks_per_grid_x = (N + threads_per_block_x - 1) / threads_per_block_x;
    int blocks_per_grid_y = (M + threads_per_block_y - 1) / threads_per_block_y;

    dim3 block_size(threads_per_block_x, threads_per_block_y);
    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y);

    sigmoid_kernel<<<num_blocks, block_size>>>(A, B, M, N);

    return 0;
}

/* 
GPU kernel for softmax. Each thread calculates one element.
*/
__global__
void softmax_kernel(double* A, double *B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // Sum up exponentials for given column
        double expsum = 0;
        for (int i = 0; i < M; i++) {
            expsum += exp(A[col * M +i]);
        }

        // Calculate softmax for given (row, column)
        B[col * M + row] = exp(A[col * M + row]) / expsum;
    }
}

/* 
Routine to perform element-wise softmax.
*/
int mySoftmax(double* A, double*B, int M, int N) {
    int threads_per_block_x = THREADS_X;
    int threads_per_block_y = THREADS_Y;
    int blocks_per_grid_x = (N + threads_per_block_x - 1) / threads_per_block_x;
    int blocks_per_grid_y = (M + threads_per_block_y - 1) / threads_per_block_y;

    dim3 block_size(threads_per_block_x, threads_per_block_y);
    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y);

    softmax_kernel<<<num_blocks, block_size>>>(A, B, M, N);

    return 0;
}

/* 
GPU kernel for repmat. Each thread broadcasts one row of b.
*/
__global__
void repmat_kernel(double* b, double* B, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        for (int j = 0; j < N; j++) {
            B[j * M + row] = b[row];
        }
    }
}

/* 
Routine to broadcast vector b of dim M to B of dim MxN.
*/
int myRepMat(double* b, double* B, int M, int N) {
    int block_size = 1024;
    int num_blocks = (M + block_size - 1) / block_size;

    repmat_kernel<<<num_blocks, block_size>>>(b, B, M, N);

    return 0;
}

/* 
GPU kernel for to add/subtract two matrices. Each thread adds one elements.
*/
__global__
void addmat_kernel(double* A, double* B, double* C, double alpha, double beta,
                   int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = col * M + row;

    if (row < M && col < N) {
        C[idx] = alpha * A[idx] + beta * B[idx];
    } 
}

/* 
Routine to add two matrices A and B, each multiplicted by a scalar factor. Result stored in C.
*/
int myAddMat(double* A, double* B, double* C, double* alpha, double* beta,
             int M, int N) {
    int threads_per_block_x = THREADS_X;
    int threads_per_block_y = THREADS_Y;
    int blocks_per_grid_x = (N + threads_per_block_x - 1) / threads_per_block_x;
    int blocks_per_grid_y = (M + threads_per_block_y - 1) / threads_per_block_y;

    dim3 block_size(threads_per_block_x, threads_per_block_y);
    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y);

    addmat_kernel<<<num_blocks, block_size>>>(A, B, C, *alpha, *beta, M, N);

    return 0;
}

/* 
GPU kernel for summing along column of matrix. Each thread sums one row.
*/
__global__
void sumcol_kernel(double* A, double* a, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // Sum up across column for given row
        double sum_col = 0;
        for (int j = 0; j < N; j++) {
            sum_col += A[j * M + row];
        }
        a[row] = sum_col;
    }
}

/*
Routine to sum across column of a matrix.
*/
int mySumCol(double* A, double* a, int M, int N) {
    int threads_per_block_x = THREADS_X;
    int threads_per_block_y = THREADS_Y;
    int blocks_per_grid_x = (N + threads_per_block_x - 1) / threads_per_block_x;
    int blocks_per_grid_y = (M + threads_per_block_y - 1) / threads_per_block_y;

    dim3 block_size(threads_per_block_x, threads_per_block_y);
    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y);

    sumcol_kernel<<<num_blocks, block_size>>>(A, a, M, N);

    return 0;
}

/* 
GPU kernel for calculating one minus matrix A.
*/
__global__
void oneminusmat_kernel(double* A, double* B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = col * M + row;

    if (row < M && col < N) {
        B[idx] = 1.0 - A[idx];
    }
}

/*
Routine to calculate matrix of all ones minus matrix A.
*/
int myOneMinusMat(double* A, double* B, int M, int N) {
    int threads_per_block_x = THREADS_X;
    int threads_per_block_y = THREADS_Y;
    int blocks_per_grid_x = (N + threads_per_block_x - 1) / threads_per_block_x;
    int blocks_per_grid_y = (M + threads_per_block_y - 1) / threads_per_block_y;

    dim3 block_size(threads_per_block_x, threads_per_block_y);
    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y);

    oneminusmat_kernel<<<num_blocks, block_size>>>(A, B, M, N);

    return 0;
}


/* 
GPU kernel for elementwise product. Each thread multiplies one entry.
*/
__global__
void elemprod_kernel(double* A, double* B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = col * M + row;

    if (row < M && col < N) {
        B[idx] *= A[idx];
    } 
}

/* 
Routine to multiply two matrices A and B in-place. Result stored in B.
*/
int myElemProd(double* A, double* B, int M, int N) {
    int threads_per_block_x = THREADS_X;
    int threads_per_block_y = THREADS_Y;
    int blocks_per_grid_x = (N + threads_per_block_x - 1) / threads_per_block_x;
    int blocks_per_grid_y = (M + threads_per_block_y - 1) / threads_per_block_y;

    dim3 block_size(threads_per_block_x, threads_per_block_y);
    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y);

    elemprod_kernel<<<num_blocks, block_size>>>(A, B, M, N);

    return 0;
}

/* 
GPU kernel for matrix tranpose product. Each thread multiplies one entry.
*/
__global__
void tranpose_kernel(double* A, double* T, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = col * M + row;
    int t_idx = row * N + col;

    if (row < M && col < N) {
        T[t_idx] = A[idx];
    } 
}

/* 
Routine to tranpose matrix A and store result in matrix T.
*/
int myTranspose(double* A, double* T, int M, int N) {
    int threads_per_block_x = THREADS_X;
    int threads_per_block_y = THREADS_Y;
    int blocks_per_grid_x = (N + threads_per_block_x - 1) / threads_per_block_x;
    int blocks_per_grid_y = (M + threads_per_block_y - 1) / threads_per_block_y;

    dim3 block_size(threads_per_block_x, threads_per_block_y);
    dim3 num_blocks(blocks_per_grid_x, blocks_per_grid_y);

    tranpose_kernel<<<num_blocks, block_size>>>(A, T, M, N);

    return 0;
}
