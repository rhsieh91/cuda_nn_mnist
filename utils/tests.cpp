#include "tests.h"
#include "gpu_func.h"
#include "cublas_v2.h"
#include "mpi.h"
#include "common.h"

#include<iomanip>
using namespace std;

#define SCALE 1         // Factor to SCALE the GEMM problem size by
#define NUM_ITERS 10    // Number of GEMMs run for timing purposes
#define GEMM_TOL 1e-12  // Tolerance for GEMM comparison

// check whether the matrix from Seq is the same as from Par.
// write out mismathces to a file.
int checkErrors(const arma::mat& Seq, const arma::mat& Par,
                std::ofstream& ofs, std::vector<double>& errors) {
    //std::ofstream ofs(filename.c_str());
    int error = 0;

    for(int i = 0; i < Seq.n_rows; ++i) {
        for(int j = 0; j < Seq.n_cols; ++j) {
            if(abs(Seq(i,j) - Par(i,j)) > 1e-4) {
                ofs << "Mismatch at pos (" << i << ", " << j << " )seq: "
                    << Seq(i,j) << " par: " << Par(i,j) << endl;
                ++error;
            }
        }
    }

    if(error)
        ofs << "There were " << error <<
            " total locations where there was a difference between the seq and par" << endl;

    double err_max = arma::norm(Seq-Par,"inf")/arma::norm(Seq,"inf");
    double err_l2  = arma::norm(Seq-Par,2)/arma::norm(Seq,2);

    if(err_max > 1e-7) {
        cout << "Correctness test failed" << endl;
    }

    errors.push_back(err_max);
    errors.push_back(err_l2);

    //ofs.close();

    return error;
}

int checkNNErrors(NeuralNetwork& seq_nn, NeuralNetwork& par_nn,
                  std::string filename) {
    std::vector<double> errors_w, errors_b;
    int error = 0;
    std::ofstream ofs(filename.c_str());

    cout << endl;

    for(int i = 0; i < seq_nn.num_layers; i++) {
        ofs << "Mismatches for W[" << i << "]" << endl;
        error += checkErrors(seq_nn.W[i], par_nn.W[i], ofs, errors_w);
        ofs << "Mismatches for b[" << i << "]" << endl;
        error += checkErrors(seq_nn.b[i], par_nn.b[i], ofs, errors_b);
        cout << endl;
        cout << "max norm of diff b/w seq and par: W[" << i << "]: " << setprecision(
                 6) << errors_w[2*i]
             << ", b[" << i << "]: " << errors_b[2*i] << endl;
        cout << "l2  norm of diff b/w seq and par: W[" << i << "]: " << setprecision(
                 6) << errors_w[2*i+1]
             << ", b[" << i << "]: " << errors_b[2*i+1] << endl;
    }

    ofs.close();
    return error;
}

void createMATS(double* A, double* B, double* C1, double* C2, int NI, int NJ,
                int NK) {
    int i, j;

    for(j = 0; j < NK; j++) {
        for(i = 0; i < NI; i++) {
            A[i + j*NI] = ((double) i*j) / NI;
        }
    }

    for(j = 0; j < NJ; j++) {
        for(i = 0; i < NK; i++) {
            B[i + j*NK] = ((double) i*j + 1) / NJ;
        }
    }

    for(j = 0; j < NJ; j++) {
        for(i = 0; i < NI; i++) {
            C1[i + j*NI] = 0;
            C2[i + j*NI] = ((double) i*j + 2) / NJ;
        }
    }
}

int compareGEMMResults(double* myC, double* refC, int NI, int NJ) {
    int i, j;
    int fail = 0;

    arma::mat mysol = arma::mat(myC, NI, NJ, false);
    arma::mat refsol = arma::mat(refC, NI, NJ, false);

    double reldiff = arma::norm(mysol-refsol,"inf")/arma::norm(refsol,"inf");

    if(reldiff > GEMM_TOL) {
        fail = 1;
    }

    // Print results
    if(fail) {
        std::cout << "My GEMM output not matching with reference. Rel diff = "
                  << reldiff << std::endl;
    } else {
        std::cout << "GEMM matched with reference successfully! Rel diff = "
                  << reldiff << std::endl;
    }

    return fail;
}

void TestGEMM(int M, int N, int K) {

    double* A;
    double* B;
    double* C1;
    double* C2;

    double* dA;
    double* dB;
    double* dC1;
    double* dC2;
    double* dummy;

    double alpha = 2.0;
    double beta = 5.0;

    int num_iters = 100;

    A = (double*)malloc(M*K*sizeof(double));
    B = (double*)malloc(K*N*sizeof(double));
    C1 = (double*)malloc(M*N*sizeof(double));
    C2 = (double*)malloc(M*N*sizeof(double));

    cudaMalloc((void**)&dA, sizeof(double) * M * K);
    cudaMalloc((void**)&dB, sizeof(double) * K * N);
    cudaMalloc((void**)&dC1, sizeof(double) * M * N);
    cudaMalloc((void**)&dC2, sizeof(double) * M * N);
    cudaMalloc((void**)&dummy, sizeof(double) * M * N);

    // C1 and C2 are same. We just have two copies to compare results
    createMATS(A, B, C1, C2, M, N, K);

    cudaMemcpy(dA, A, sizeof(double) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(double) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dC1, C2, sizeof(double) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dC2, C2, sizeof(double) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dummy, C2, sizeof(double) * M * N, cudaMemcpyHostToDevice);


    /* Warm up GPU before we run. We run one extra CuBlas */
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);

    if(stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization failed!" << std::endl;
        return;
    }

    stat = cublasDgemm(handle,
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       M, N, K,
                       &alpha,
                       dA, M,
                       dB, K,
                       &beta,
                       dummy, M);

    /* Compute reference solution and time the CuBlas */
    double refstart = MPI_Wtime();

    for(int i = 0; i < NUM_ITERS; i++) {
        stat = cublasDgemm(handle,
                           CUBLAS_OP_N, CUBLAS_OP_N,
                           M, N, K,
                           &alpha,
                           dA, M,
                           dB, K,
                           &beta,
                           dC2, M);
    }

    check_launch("Reference GEMM");
    double refend = MPI_Wtime();

    if(stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS gemm error at " << __FILE__ << ":" << __LINE__ <<
                  std::endl;
    }

    cudaMemcpy(C2, dC2, sizeof(double) * M * N, cudaMemcpyDeviceToHost);

    /* We are calling your GEMM function here */
    /* We will make one dummy call and check_launch here */
    int err;
    err = myGEMM(dA, dB, dummy, &alpha, &beta, M, N, K);
    check_launch("myGEMM dummy");

    double mystart = MPI_Wtime();

    for(int i = 0; i < NUM_ITERS; i++) {
        err = myGEMM(dA, dB, dC1, &alpha, &beta, M, N, K);
    }

    check_launch("myGEMM");
    double myend = MPI_Wtime();


    /* This error code is for your own debugging, it does not catch
       illegal memory accesses or bad kernel launches */
    if(err!=0) {
        std::cout << "Error in my GEMM. Error code: " << err << std::endl;
    }

    cudaMemcpy(C1, dC1, sizeof(double) * M * N, cudaMemcpyDeviceToHost);

    int fail = compareGEMMResults(C1, C2, M, N);

    if(fail == 0) {
        std::cout << "Time for reference GEMM implementation: "
                  << refend - refstart << std::endl;
        std::cout << "Time for my GEMM implementation: "
                  << myend - mystart << std::endl;
    }

    free(A);
    free(B);
    free(C1);
    free(C2);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC1);
    cudaFree(dC2);
    cudaFree(dummy);
}

void BenchmarkGEMM() {

    std::cout << std::endl << "Entering GEMM Benchmarking mode! Stand by."
              << std::endl;

    /* First GEMM Problem Size */
    int M = 800*SCALE, N = 1000*SCALE, K = 784*SCALE;

    std::cout << std::endl << "Starting GEMM 1: " << "M = " << M << "; N = "
              << N << "; K = " << K << std::endl;
    TestGEMM(M, N, K);
    std::cout << "Completed GEMM 1" << std::endl;

    /* Secong GEMM Problem Size */
    M = 800*SCALE, N = 10*SCALE, K = 1000*SCALE;
    std::cout << std::endl << "Starting GEMM 2: " << "M = " << M << "; N = "
              << N << "; K = " << K << std::endl;
    TestGEMM(M, N, K);
    std::cout << "Completed GEMM 2" << std::endl;
}


void create_mat(double* A, int M, int N) {
    int row, col;

    for(col = 0; col < N; col++) {
        for(row = 0; row < M; row++) {
            A[col * M + row] = ((double) row*col) / M;
        }
    }
}

void create_mat_2(double* A, int M, int N) {
    int row, col;

    for(col = 0; col < N; col++) {
        for(row = 0; row < M; row++) {
            A[col * M + row] = (double) col * M + row;
        }
    }
}

void create_armamat(arma::mat &A) {
    for (int i = 0; i < A.n_rows; i++) {
        for (int j = 0; j < A.n_cols; j++) {
            A(i, j) = ((double) i*j) / A.n_rows;
        }
    }
}

void create_vec(double* b, int M) {
    int row;

    for(row = 0; row < M; row++) {
        b[row] = ((double) row) / M;
    }
}

void TestSigmoid(int M, int N) {
    double* Z2;
    double* A2;
    double* dZ2;
    double* dA2;

    Z2 = (double*)malloc(M*N*sizeof(double));
    A2 = (double*)malloc(M*N*sizeof(double));
    cudaMalloc((void**)&dZ2, M*N*sizeof(double));
    cudaMalloc((void**)&dA2, M*N*sizeof(double));

    create_mat(Z2, M, N);

    cudaMemcpy(dZ2, Z2, M*N*sizeof(double), cudaMemcpyHostToDevice);

    // CPU solution
    arma::mat Z1(M, N), A1(M, N);
    create_armamat(Z1);
    sigmoid(Z1, A1);

    // GPU solution
    int err;
    err = mySigmoid(dZ2, dA2, M, N);
    check_launch("mySigmoid");

    if(err!=0) {
        std::cout << "Error in mySigmoid. Error code: " << err << std::endl;
    }

    cudaMemcpy(A2, dA2, M*N*sizeof(double), cudaMemcpyDeviceToHost);

    // Print CPU/GPU results
    std::cout << "CPU solution: " << std::endl;
    A1.print();
    std::cout << "GPU solution: " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A2[j * M + i] << "   ";
        }  
        std::cout << std::endl;
    }

    free(Z2);
    free(A2);
    cudaFree(dZ2);
    cudaFree(dA2);
}

void TestSoftmax(int M, int N) {
    double* Z2;
    double* A2;
    double* dZ2;
    double* dA2;

    Z2 = (double*)malloc(M*N*sizeof(double));
    A2 = (double*)malloc(M*N*sizeof(double));
    cudaMalloc((void**)&dZ2, M*N*sizeof(double));
    cudaMalloc((void**)&dA2, M*N*sizeof(double));

    create_mat(Z2, M, N);

    cudaMemcpy(dZ2, Z2, M*N*sizeof(double), cudaMemcpyHostToDevice);

    // CPU solution
    arma::mat Z1(M, N), A1(M, N);
    create_armamat(Z1);
    softmax(Z1, A1);

    // GPU solution
    int err;
    err = mySoftmax(dZ2, dA2, M, N);
    check_launch("mySoftmax");

    if(err!=0) {
        std::cout << "Error in mySoftmax. Error code: " << err << std::endl;
    }

    cudaMemcpy(A2, dA2, M*N*sizeof(double), cudaMemcpyDeviceToHost);

    // Print CPU/GPU results
    std::cout << "CPU solution: " << std::endl;
    A1.print();
    std::cout << "GPU solution: " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A2[j * M + i] << "  ";
        }  
        std::cout << std::endl;
    }

    free(Z2);
    free(A2);
    cudaFree(dZ2);
    cudaFree(dA2);
}

void TestRepMat(int M, int N) {
    double* b;
    double* B;
    double* db;
    double* dB;

    b = (double*)malloc(M*sizeof(double));
    B = (double*)malloc(M*N*sizeof(double));

    cudaMalloc((void**)&db, M*N*sizeof(double));
    cudaMalloc((void**)&dB, M*N*sizeof(double));

    create_vec(b, M);

    cudaMemcpy(db, b, M*sizeof(double), cudaMemcpyHostToDevice);

    int err;
    err = myRepMat(db, dB, M, N);
    check_launch("myRepMat");

    cudaMemcpy(B, dB, M*N*sizeof(double), cudaMemcpyDeviceToHost);

    if(err!=0) {
        std::cout << "Error in myRepMat. Error code: " << err << std::endl;
    }

    std::cout << "b = " << std::endl;
    for (int i = 0; i < M; i++) {
        std::cout << b[i] << std::endl; 
    }


    std::cout << "B = " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << B[j * M + i] << "  ";
        }  
        std::cout << std::endl;
    }

    free(b);
    free(B);
    cudaFree(db);
    cudaFree(dB);
}

void TestAddMat(int M, int N) {
    double* A;
    double* B; 
    double* C;
    double* dA;
    double* dB;
    double* dC;

    double alpha = 2.0;
    double beta = -1.0;

    A = (double*)malloc(M*N*sizeof(double));
    B = (double*)malloc(M*N*sizeof(double));
    C = (double*)malloc(M*N*sizeof(double));

    cudaMalloc((void**)&dA, M*N*sizeof(double));
    cudaMalloc((void**)&dB, M*N*sizeof(double));
    cudaMalloc((void**)&dC, M*N*sizeof(double));

    create_mat(A, M, N);
    create_mat(B, M, N);

    cudaMemcpy(dA, A, M*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, M*N*sizeof(double), cudaMemcpyHostToDevice);

    int err;
    err = myAddMat(dA, dB, dC, &alpha, &beta, M, N);
    check_launch("myAddMat");

    if(err!=0) {
        std::cout << "Error in myAddMat. Error code: " << err << std::endl;
    }

    cudaMemcpy(C, dC, M*N*sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "A: " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A[j * M + i] << "   ";
        }  
        std::cout << std::endl;
    }
    std::cout << "B: " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << B[j * M + i] << "   ";
        }  
        std::cout << std::endl;
    }
    std::cout << "C: " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[j * M + i] << "   ";
        }  
        std::cout << std::endl;
    }

    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

void TestSumCol(int M, int N) {
    double* A;
    double* a;
    double* dA;
    double* da;

    A = (double*)malloc(M*N*sizeof(double));
    a = (double*)malloc(M*sizeof(double));
    cudaMalloc((void**)&dA, M*N*sizeof(double));
    cudaMalloc((void**)&da, M*sizeof(double));

    create_mat(A, M, N);

    cudaMemcpy(dA, A, M*N*sizeof(double), cudaMemcpyHostToDevice);

    int err;
    err = mySumCol(dA, da, M, N);
    check_launch("mySumCol");

    if(err!=0) {
        std::cout << "Error in mySumCol. Error code: " << err << std::endl;
    }

    cudaMemcpy(a, da, M*sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "GPU solution: " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A[j * M + i] << "  ";
        }  
        std::cout << std::endl;
    }
    for (int i = 0; i < M; i++) {
        std::cout << a[i] << " ";
         std::cout << std::endl;
    }

    free(A);
    free(a);
    cudaFree(dA);
    cudaFree(da);
}

void TestOneMinusMat(int M, int N) {
    double* A;
    double* B;
    double* dA;
    double* dB;

    A = (double*)malloc(M*N*sizeof(double));
    B = (double*)malloc(M*N*sizeof(double));

    cudaMalloc((void**)&dA, M*N*sizeof(double));
    cudaMalloc((void**)&dB, M*N*sizeof(double));

    create_mat(A, M, N);

    cudaMemcpy(dA, A, M*N*sizeof(double), cudaMemcpyHostToDevice);

    int err;
    err = myOneMinusMat(dA, dB, M, N);
    check_launch("myOneMinusMat");

    if(err!=0) {
        std::cout << "Error in myOneMinusMat. Error code: " << err << std::endl;
    }

    cudaMemcpy(B, dB, M*N*sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "A: " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A[j * M + i] << "   ";
        }  
        std::cout << std::endl;
    }
    std::cout << "B: " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << B[j * M + i] << "   ";
        }  
        std::cout << std::endl;
    }

    free(A);
    free(B);
    cudaFree(dA);
    cudaFree(dB);
}

void TestElemProd(int M, int N) {
    double* A;
    double* B; 
    // double* C;
    double* dA;
    double* dB;
    // double* dC;

    A = (double*)malloc(M*N*sizeof(double));
    B = (double*)malloc(M*N*sizeof(double));
    // C = (double*)malloc(M*N*sizeof(double));

    cudaMalloc((void**)&dA, M*N*sizeof(double));
    cudaMalloc((void**)&dB, M*N*sizeof(double));
    // cudaMalloc((void**)&dC, M*N*sizeof(double));

    create_mat(A, M, N);
    create_mat(B, M, N);

    cudaMemcpy(dA, A, M*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, M*N*sizeof(double), cudaMemcpyHostToDevice);

    int err;
    err = myElemProd(dA, dB, M, N);
    check_launch("myElemProd");

    if(err!=0) {
        std::cout << "Error in myElemProd. Error code: " << err << std::endl;
    }

    cudaMemcpy(B, dB, M*N*sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "A: " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A[j * M + i] << "   ";
        }  
        std::cout << std::endl;
    }
    std::cout << "B: " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << B[j * M + i] << "   ";
        }  
        std::cout << std::endl;
    }
}

void TestTranspose(int M, int N) {
    double* A;
    double* T; 
    double* dA;
    double* dT;

    A = (double*)malloc(M*N*sizeof(double));
    T = (double*)malloc(N*M*sizeof(double));

    cudaMalloc((void**)&dA, M*N*sizeof(double));
    cudaMalloc((void**)&dT, N*M*sizeof(double));

    create_mat_2(A, M, N);

    cudaMemcpy(dA, A, M*N*sizeof(double), cudaMemcpyHostToDevice);

    int err;
    err = myTranspose(dA, dT, M, N);
    check_launch("myTranspose");

    if(err!=0) {
        std::cout << "Error in myTranspose. Error code: " << err << std::endl;
    }

    cudaMemcpy(T, dT, M*N*sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "A: " << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A[j * M + i] << "   ";
        }  
        std::cout << std::endl;
    }
    std::cout << "T: " << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << T[i + j * N] << "   ";
        }  
        std::cout << std::endl;
    }
}