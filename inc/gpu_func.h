#ifndef GPU_FUNC_H_
#define GPU_FUNC_H_

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

struct event_pair {
    cudaEvent_t start;
    cudaEvent_t end;
};

inline void check_launch(const char* kernel_name) {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        std::cerr << "error in " << kernel_name << " kernel" << std::endl;
        std::cerr << "error was: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

inline void start_timer(event_pair* p) {
    cudaEventCreate(&p->start);
    cudaEventCreate(&p->end);
    cudaEventRecord(p->start, 0);
}


inline double stop_timer(event_pair* p) {
    cudaEventRecord(p->end, 0);
    cudaEventSynchronize(p->end);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, p->start, p->end);
    cudaEventDestroy(p->start);
    cudaEventDestroy(p->end);
    return elapsed_time;
}

int useless_gpu_add_one(int t);

int myGEMM(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K);

int myGEMMGlobal(double* A, double* B, double* C, double* alpha, double* beta, int M,
           int N, int K);

int mySigmoid(double* A, double *B, int M, int N);

int mySoftmax(double* A,double *B,  int M, int N);

int myRepMat(double* b, double* B, int M, int N);

int myAddMat(double* A, double* B, double* C, double* alpha, double* beta, int M, int N);

int mySumCol(double* A, double* a, int M, int N);

int myOneMinusMat(double* A, double* B, int M, int N);

int myElemProd(double* A, double* B, int M, int N);

int myTranspose(double* A, double* T, int M, int N);

#endif
