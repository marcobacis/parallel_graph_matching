/*

Functions to implement

- read_graph -> read sparse Graph
- decomposition -> decomposes H using SVD
- compute_tilde -> to compute A,B (in the initialization phase)
- compute_w_iterate/compute_z_iterate
- compute_X_iterate

*/


#include <fstream>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <boost/numeric/ublas/storage.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>

namespace ublas = boost::numeric::ublas;

typedef ublas::mapped_matrix<float > matrix_t;
typedef ublas::vector<float> vector_t;

typedef matrix_t::iterator1 i1_t;
typedef matrix_t::iterator2 i2_t;

matrix_t readMtxFile(std::string filename);

matrix_t compute_trans(matrix_t mat);

matrix_t compute_norm(matrix_t mat);

vector_t matvect_prod(matrix_t mat, vector_t vect);
