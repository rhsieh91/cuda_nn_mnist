# 2-layer Neural Network using CUDA &amp; MPI
This repository contains the final project for Stanford's course in Parallel Computing with CUDA and MPI (CME213). Specifically, a GPU-accelerated parallel implementation of a 2-layer neural network was benchmarked against a pure serial implementation in CPU. 
* All training and testing were performed on the MNIST dataset.
* CUDA kernels were written to enable GPU-accelerated matrix operations in the forward and backward pass of the neural network.
* MPI was used in the training and testing phases to distribute data across multiple GPUs.

## Starter Code
The following were provided as starter code by the teaching staff of CME213:
* utils/common.cpp
* utils/common.h
* utils/mnist.cpp
* utils/mnist.h
* serial baseline in neural_network.cpp
* functions for comparing neural networks and benchmarking general matrix multiply (GEMM) in utils/test.cpp

## My Implementation
The following were implemented by myself:
* gpu_func.cu
* gpu_func.h
* parellel implementation in neural_network.cpp
* functions for testing other CUDA kernels in utils/test.cpp (e.g. matrix transrpose, softmax, etc.)

## Compiling the Code
This project was compiled on a virtual machine (VM) in Google Cloud. For ease of use, bash scripts were provided by the teaching staff to initialize the VMs properly: _create_vm_final_project*.sh_. Notable requirements are:
* make
* gcc
* CUDA
* openmpi
* cmake
* nvvp
* Armadillo (C++ linear algebra library)

A _Makefile_ is also provided for easy compilation.

## Running the Code
To run the compiled code with a single process and GPU:
```
./main [args]
```
To run the compiled code with *N* processes and GPUs
```
mpirun -mca btl ^openib -n [N] ./main [args]
```
Args:
* **-n num** controls number of neurons in the hidden layer
* **-r num** controls L2 regularization
* **-l num** controls learning rate
* **-e num** controls number of epochs
* **-b num** controls batch size 
* **./main** with no args will run all default test cases

## Details and Results
Both the serial and parallel implementations were able to achieve above 90% classification accuracy on the validation set. However, the crux of this project was to investigate the speed-up provided by a parallel implementation. By using 4 MPI processes distributed across 4 GPUs, the training time was 10x faster than the serial baseline.

Furthermore, an in-depth profiling of the CUDA kernels using NVIDIA Visual Profiler reveals that the bottleneck in the neural network is in the general matrix multiply (GEMM) operation. Two variants of GEMM were written: a straightforward implementation using global memory and an optimized implementation that used shared memory for memory re-use. As expected, the optimized GEMM kernel using shared memory was significantly faster.

