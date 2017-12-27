/*

Functions to implement

- read_graph -> read sparse Graph
- decomposition -> decomposes H using SVD
- compute_tilde -> to compute A,B (in the initialization phase)
- compute_w_iterate/compute_z_iterate
- compute_X_iterate

*/

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <omp.h>
#include <mpi.h>
#include <tuple>
#include <boost/numeric/ublas/storage.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>

namespace ublas = boost::numeric::ublas;

//Types definitions

typedef ublas::compressed_matrix<float> sparse_t;
typedef ublas::matrix<float> matrix_t;
typedef ublas::vector<float> vector_t;

typedef ublas::matrix<float>::iterator1 i1_t;
typedef ublas::matrix<float>::iterator2 i2_t;

typedef std::tuple<int, int, float> matelem;


//MPI message tags definitions
#define MSG_MATRIX_SIZE 1
#define MSG_MATRIX_X 2
#define MSG_MATRIX_Y 3
#define MSG_MATRIX_VALS 4
#define MSG_VECTOR_Z 5
#define MSG_VECTOR_W 6

//Functions prototypes

matrix_t readMtxFile(std::string filename);

matrix_t compute_trans(matrix_t mat);

matrix_t compute_norm(matrix_t mat);

vector_t matvect_prod(matrix_t mat, vector_t vect);

matrix_t compute_x_iterate(matrix_t A, matrix_t B, vector_t Z, vector_t W, int s, int n,float alpha);

void decompose_matrix(matrix_t mat, int components, std::vector<int> xs[], std::vector<int> ys[], std::vector<float> vals[], int nnz[]);

void scatter_matrix(matrix_t mat, int nproc);

void broadcast_matrix(matrix_t mat, int nproc);

matrix_t receive_matrix();
