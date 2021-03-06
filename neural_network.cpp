#include "neural_network.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

double norms(NeuralNetwork& nn) {
    double norm_sum = 0;

    for(int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu(arma::square(nn.W[i]));
    }

    return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    nn.W[0].save(s.str(), arma::raw_ascii);
    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    nn.W[1].save(t.str(), arma::raw_ascii);
    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    nn.b[0].save(u.str(), arma::raw_ascii);
    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
    arma::mat A, B, C, D;

    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    A.load(s.str(), arma::raw_ascii);
    double max_errW0 = arma::norm(nn.W[0]-A, "inf")/arma::norm(A, "inf");
    double L2_errW0  = arma::norm(nn.W[0]-A,2)/arma::norm(A,2);

    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    B.load(t.str(), arma::raw_ascii);
    double max_errW1 = arma::norm(nn.W[1]-B, "inf")/arma::norm(B, "inf");
    double L2_errW1  = arma::norm(nn.W[1]-B,2)/arma::norm(B,2);

    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    C.load(u.str(), arma::raw_ascii);
    double max_errb0 = arma::norm(nn.b[0]-C, "inf")/arma::norm(C, "inf");
    double L2_errb0  = arma::norm(nn.b[0]-C,2)/arma::norm(C,2);

    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    D.load(v.str(), arma::raw_ascii);
    double max_errb1 = arma::norm(nn.b[1]-D, "inf")/arma::norm(D, "inf");
    double L2_errb1  = arma::norm(nn.b[1]-D,2)/arma::norm(D,2);

    int ow = 15;

    if(iter == 0) {
        error_file << std::left<< std::setw(ow) << "Iteration" << std::left<< std::setw(
                       ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1"
                   << std::left<< std::setw(ow) << "Max Err b0" << std::left<< std::setw(
                       ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left
                   << std::setw(ow) << "L2 Err W1" << std::left<< std::setw(
                       ow) << "L2 Err b0" << std::left<< std::setw(ow) << "L2 Err b1" << "\n";
    }

    error_file << std::left << std::setw(ow) << iter << std::left << std::setw(
                   ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 <<
               std::left << std::setw(ow) << max_errb0 << std::left << std::setw(
                   ow) << max_errb1 << std::left<< std::setw(ow) << L2_errW0 << std::left <<
               std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 <<
               std::left<< std::setw(ow) << L2_errb1 << "\n";

}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache) {
    cache.z.resize(2);
    cache.a.resize(2);

    // std::cout << W[0].n_rows << "\n";tw
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    arma::mat z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
    cache.z[0] = z1;

    arma::mat a1;
    sigmoid(z1, a1);
    cache.a[0] = a1;

    assert(a1.n_rows == nn.W[1].n_cols);
    arma::mat z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
    cache.z[1] = z2;

    arma::mat a2;
    softmax(z2, a2);
    cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_cols;

    // std::cout << "backprop " << bpcache.yc << "\n";
    arma::mat diff = (1.0 / N) * (bpcache.yc - y);

    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    bpgrads.db[1] = arma::sum(diff, 1);
    arma::mat da1 = nn.W[1].t() * diff;

    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss(NeuralNetwork& nn, const arma::mat& yc, const arma::mat& y,
            double reg) {
    int N = yc.n_cols;
    double ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

    double data_loss = ce_sum / N;
    double reg_loss = 0.5 * reg * norms(nn);
    double loss = data_loss + reg_loss;
    // std::cout << "Loss: " << loss << "\n";
    return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::mat& X, arma::rowvec& label) {
    struct cache fcache;
    feedforward(nn, X, fcache);
    label.set_size(X.n_cols);

    for(int i = 0; i < X.n_cols; ++i) {
        arma::uword row;
        fcache.yc.col(i).max(row);
        label(i) = row;
    }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
             double reg, struct grads& numgrads) {
    double h = 0.00001;
    struct cache numcache;
    numgrads.dW.resize(nn.num_layers);
    numgrads.db.resize(nn.num_layers);

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

        for(int j = 0; j < nn.W[i].n_rows; ++j) {
            for(int k = 0; k < nn.W[i].n_cols; ++k) {
                double oldval = nn.W[i](j,k);
                nn.W[i](j, k) = oldval + h;
                feedforward(nn, X, numcache);
                double fxph = loss(nn, numcache.yc, y, reg);
                nn.W[i](j, k) = oldval - h;
                feedforward(nn, X, numcache);
                double fxnh = loss(nn, numcache.yc, y, reg);
                numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
                nn.W[i](j, k) = oldval;
            }
        }
    }

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

        for(int j = 0; j < nn.b[i].size(); ++j) {
            double oldval = nn.b[i](j);
            nn.b[i](j) = oldval + h;
            feedforward(nn, X, numcache);
            double fxph = loss(nn, numcache.yc, y, reg);
            nn.b[i](j) = oldval - h;
            feedforward(nn, X, numcache);
            double fxnh = loss(nn, numcache.yc, y, reg);
            numgrads.db[i](j) = (fxph - fxnh) / (2*h);
            nn.b[i](j) = oldval;
        }
    }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
           double learning_rate, double reg,
           const int epochs, const int batch_size, bool grad_check, int print_every,
           int debug) {
    int N = X.n_cols;
    std::cout << "N: " << N << std::endl;
    int iter = 0;
    int print_flag = 0;

    for(int epoch = 0 ; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

        for(int batch = 0; batch < num_batches; ++batch) {
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;
            feedforward(nn, X_batch, bpcache);

            struct grads bpgrads;
            backprop(nn, y_batch, reg, bpcache, bpgrads);

            if(print_every > 0 && iter % print_every == 0) {
                if(grad_check) {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
            }

            // Gradient descent step
            for(int i = 0; i < nn.W.size(); ++i) {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for(int i = 0; i < nn.b.size(); ++i) {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }

            /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
               for the first batch of each epoch to avoid saving too many large files.
               Note that for the first time, you have to run debug and serial modes together.
               This will run the following function and write out files to CPUmats folder.
               In the later runs (with same parameters), you can use just the debug flag to
               output diff b/w CPU and GPU without running CPU version */
            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            if(debug && print_flag) {
                write_cpudata_tofile(nn, iter);
            }

            iter++;
        }
    }
}

// ************************************************************************
// BEGINNING OF PARALLEL IMPLEMENTATION
// ************************************************************************
class NNDeviceCache {
public:
    double *W1, *b1, *W2, *b2; // network weights and biases
    double *Z1, *A1, *Z2, *YC; // hidden layer results
    double *diff; // YC - Y
    double *dW1, *db1, *dW2, *db2; // gradients
    double *dZ1, *dA1;
    double *A1T, *W2T, *XT; // tranposes for backprop

    NNDeviceCache (NeuralNetwork& nn, int H, int d, int C, int N) {
        cudaMalloc((void**)&W1, H*d*sizeof(double));
        cudaMalloc((void**)&b1, H*sizeof(double));
        cudaMalloc((void**)&W2, C*H*sizeof(double));
        cudaMalloc((void**)&b2, C*sizeof(double));
        cudaMalloc((void**)&Z1, H*N*sizeof(double));
        cudaMalloc((void**)&A1, H*N*sizeof(double));
        cudaMalloc((void**)&Z2, C*N*sizeof(double));
        cudaMalloc((void**)&YC, C*N*sizeof(double));
        cudaMalloc((void**)&diff, C*N*sizeof(double));
        cudaMalloc((void**)&dW1, H*d*sizeof(double));
        cudaMalloc((void**)&db1, H*sizeof(double));
        cudaMalloc((void**)&dW2, C*H*sizeof(double));
        cudaMalloc((void**)&db2, C*sizeof(double));
        cudaMalloc((void**)&dZ1, H*N*sizeof(double));
        cudaMalloc((void**)&dA1, H*N*sizeof(double));
        cudaMalloc((void**)&A1T, N*H*sizeof(double));
        cudaMalloc((void**)&W2T, H*C*sizeof(double));
        cudaMalloc((void**)&XT, N*d*sizeof(double));
    }

    ~NNDeviceCache() {
        cudaFree(W1);
        cudaFree(b1);
        cudaFree(W2);
        cudaFree(b2);
        cudaFree(Z1);
        cudaFree(A1);
        cudaFree(Z2);
        cudaFree(YC);
        cudaFree(diff);
        cudaFree(dW1);
        cudaFree(db1);
        cudaFree(dW2);
        cudaFree(db2);
        cudaFree(dZ1);
        cudaFree(dA1);
        cudaFree(A1T);
        cudaFree(W2T);
        cudaFree(XT);
    }
};

void gpu_feedforward(NeuralNetwork& nn, NNDeviceCache &cache, double* dX, int N) {
    int d = nn.H[0]; // num features 
    int H = nn.H[1]; // num neurons in hidden layer
    int C = nn.H[2]; // num output classes
    double alpha = 1.0;
    double beta = 1.0;

    myRepMat(cache.b1, cache.Z1, H, N);

    myGEMM(cache.W1, dX, cache.Z1, &alpha, &beta, H, N, d);

    mySigmoid(cache.Z1, cache.A1, H, N);

    myRepMat(cache.b2, cache.Z2, C, N);
    myGEMM(cache.W2, cache.A1, cache.Z2, &alpha, &beta, C, N, H);

    mySoftmax(cache.Z2, cache.YC, C, N);
}

void gpu_backprop(NeuralNetwork& nn, NNDeviceCache &cache, double* dX, double* dY, 
                  double reg, int minibatch_size, int subbatch_size) {
    int d = nn.H[0]; // num features 
    int H = nn.H[1]; // num neurons in hidden layer
    int C = nn.H[2]; // num output classes
    double one = 1.0;
    double pos_norm = 1.0 / minibatch_size;
    double neg_norm = -1.0 / minibatch_size;
    double zero = 0.0;

    myAddMat(cache.YC, dY, cache.diff, &pos_norm, &neg_norm, C, subbatch_size);

    myTranspose(cache.A1, cache.A1T, H, subbatch_size);
    myGEMM(cache.diff, cache.A1T, cache.dW2, &one, &reg, C, H, subbatch_size); // W2 is copied to dW2 when cache is initialized

    mySumCol(cache.diff, cache.db2, C, subbatch_size);

    myTranspose(cache.W2, cache.W2T, C, H);
    myGEMM(cache.W2T, cache.diff, cache.dA1, &one, &zero, H, subbatch_size, C);

    myOneMinusMat(cache.A1, cache.dZ1, H, subbatch_size); // 1 - A1
    myElemProd(cache.A1, cache.dZ1, H, subbatch_size); // A1 * (1 - A1)
    myElemProd(cache.dA1, cache.dZ1, H, subbatch_size); // dA1 * A1 * (1 - A1), stored in cache.dZ1
    
    myTranspose(dX, cache.XT, d, subbatch_size);
    myGEMM(cache.dZ1, cache.XT, cache.dW1, &one, &reg, H, d, subbatch_size); // W1 is copied to dW1 when cache is initialized
    
    mySumCol(cache.dZ1, cache.db1, H, subbatch_size);
}

/*
 * TODO
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0)?X.n_cols:0; // if rank=0, N=X.n_cols, else N=0
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;

    /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
       for storing elements in a column major way. Or you can allocate your own array
       memory space and store the elements in a row major way. Remember to update the
       Armadillo matrices in NeuralNetwork &nn of rank 0 before returning from the function. */

    /* iter is a variable used to manage debugging. It increments in the inner loop
       and therefore goes from 0 to epochs*num_batches */
    int iter = 0;

    int d = nn.H[0]; // num features 
    int H = nn.H[1]; // num neurons in hidden layer
    int C = nn.H[2]; // num output classes

    // Keep track of pointers to each minibatch that will be loaded to GPU
    int num_batches = (N + batch_size - 1)/batch_size;
    std::vector<double*> dX_batch(num_batches); // vector of pointers for each minibatch on GPU
    std::vector<double*> dY_batch(num_batches);

    // Iterate through minibatches, divide into smaller subbatches, and scatter
    for (int batch = 0; batch < num_batches; ++ batch) {
        int last_col = std::min((batch + 1)*batch_size-1, N-1);
        int start_col = batch * batch_size;
        int minibatch_size = last_col - start_col + 1; // dataset is divided into minibathes
        int subbatch_size = (minibatch_size + num_procs - 1) / num_procs; // each minimatch is divided into subbatches for each process
        subbatch_size = std::min(subbatch_size, minibatch_size - rank * subbatch_size);

        // Figure out displacements and counts for subbatches of given minibatch
        int *displs_X = new int[num_procs];
        int *displs_Y = new int[num_procs];
        int *counts_X = new int[num_procs];
        int *counts_Y = new int[num_procs];
        for (int i = 0; i < num_procs; ++i) {
            displs_X[i] = i * d * subbatch_size;
            displs_Y[i] = i * C * subbatch_size;
            counts_X[i] = d * subbatch_size;
            counts_Y[i] = C * subbatch_size;
        }

        // Scatter the subbatches. Allocate an arma::mat to store each subbatch.
        arma::mat X_subbatch(d, counts_X[rank] / d);
        MPI_SAFE_CALL(MPI_Scatterv(X.colptr(start_col), counts_X, displs_X,
                                   MPI_DOUBLE, X_subbatch.memptr(), counts_X[rank],
                                   MPI_DOUBLE, 0, MPI_COMM_WORLD));
        arma::mat Y_subbatch(C, counts_Y[rank] / C);
        MPI_SAFE_CALL(MPI_Scatterv(y.colptr(start_col), counts_Y, displs_Y,
                                   MPI_DOUBLE, Y_subbatch.memptr(), counts_Y[rank],
                                   MPI_DOUBLE, 0, MPI_COMM_WORLD));

        cudaMalloc((void**)&dX_batch[batch], d*subbatch_size*sizeof(double));
        cudaMalloc((void**)&dY_batch[batch], C*subbatch_size*sizeof(double));
        cudaMemcpy(dX_batch[batch], X_subbatch.memptr(), d*subbatch_size*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dY_batch[batch], Y_subbatch.memptr(), C*subbatch_size*sizeof(double), cudaMemcpyHostToDevice);

        // Free memory
        delete[] displs_X;
        delete[] displs_Y;
        delete[] counts_X;
        delete[] counts_Y;
    }

    double* sub_dW1;
    double* sub_db1;
    double* sub_dW2;
    double* sub_db2;

    sub_dW1 = (double*)malloc(H*d*sizeof(double));
    sub_db1 = (double*)malloc(H*sizeof(double));
    sub_dW2 = (double*)malloc(C*H*sizeof(double));
    sub_db2 = (double*)malloc(C*sizeof(double));

    // Training loop
    NNDeviceCache cache(nn, H, d, C, batch_size); 

    for(int epoch = 0; epoch < epochs; ++epoch) {
        for(int batch = 0; batch < num_batches; ++batch) {
            /*
             * Possible Implementation:
             * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
             * 2. compute each sub-batch of images' contribution to network coefficient updates
             * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
             * 4. update local network coefficient at each node
             */
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            int start_col = batch * batch_size;
            int minibatch_size = last_col - start_col + 1; // dataset is divided into minibathes
            int subbatch_size = (minibatch_size + num_procs - 1) / num_procs; // each minimatch is divided into subbatches for each process
            subbatch_size = std::min(subbatch_size, minibatch_size - rank * subbatch_size);

            // Copy network params from host to device
            cudaMemcpy(cache.W1, nn.W[0].memptr(), H*d*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(cache.b1, nn.b[0].memptr(), H*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(cache.W2, nn.W[1].memptr(), C*H*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(cache.b2, nn.b[1].memptr(), C*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(cache.dW1, nn.W[0].memptr(), H*d*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(cache.dW2, nn.W[1].memptr(), C*H*sizeof(double), cudaMemcpyHostToDevice);

            // Forward pass / backward pass
            gpu_feedforward(nn, cache, dX_batch[batch], subbatch_size);
            double reg_over_proc = reg / num_procs;
            gpu_backprop(nn, cache, dX_batch[batch], dY_batch[batch], reg_over_proc, minibatch_size, subbatch_size);

            // Allocate memory on the host to store gradients for each subbatch
            cudaMemcpy(sub_dW1, cache.dW1, H*d*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(sub_db1, cache.db1, H*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(sub_dW2, cache.dW2, C*H*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(sub_db2, cache.db2, C*sizeof(double), cudaMemcpyDeviceToHost);

            // Allreduce sub gradients
            arma::mat h_dW1(size(nn.W[0]), arma::fill::zeros);
            arma::mat h_db1(size(nn.b[0]), arma::fill::zeros);
            arma::mat h_dW2(size(nn.W[1]), arma::fill::zeros);
            arma::mat h_db2(size(nn.b[1]), arma::fill::zeros);

            MPI_SAFE_CALL(MPI_Allreduce(sub_dW1, h_dW1.memptr(), H*d, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(sub_db1, h_db1.memptr(), H, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(sub_dW2, h_dW2.memptr(), C*H, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
            MPI_SAFE_CALL(MPI_Allreduce(sub_db2, h_db2.memptr(), C, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
                
            nn.W[0] -= learning_rate * h_dW1;
            nn.b[0] -= learning_rate * h_db1;
            nn.W[1] -= learning_rate * h_dW2;
            nn.b[1] -= learning_rate * h_db2;

            /* Following debug routine assumes that you have already updated the arma
               matrices in the NeuralNetwork nn.  */
            if(debug && rank == 0 && print_flag) {
                write_diff_gpu_cpu(nn, iter, error_file);
            }

            iter++;
        }
    }

    // Free memory
    for (auto ptr : dX_batch) {
        cudaFree(ptr);
    }
    dX_batch.clear();

    for (auto ptr : dY_batch) {
        cudaFree(ptr);
    }
    dY_batch.clear();

    free(sub_dW1);
    free(sub_db1);
    free(sub_dW2);
    free(sub_db2);

    error_file.close();
}
