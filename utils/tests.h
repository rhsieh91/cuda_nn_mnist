#ifndef TESTS_H_
#define TESTS_H_

#include "test_utils.h"
#include "neural_network.h"

int checkErrors(const arma::mat& Seq, const arma::mat& Par,
                std::ofstream& ofs, std::vector<double>& errors);

int checkNNErrors(NeuralNetwork& seq_nn, NeuralNetwork& par_nn,
                  std::string filename);

void BenchmarkGEMM();

void TestSigmoid(int M, int N);

void TestSoftmax(int M, int N);

void TestRepMat(int M, int N);

void TestAddMat(int M, int N);

void TestSumCol(int M, int N);

void TestOneMinusMat(int M, int N);

void TestElemProd(int M, int N);

void TestTranspose(int M, int N);

#endif
