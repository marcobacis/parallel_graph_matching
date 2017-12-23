#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <ctime>
#include <iomanip>
#include "nsd.h"


void printMatrix(matrix_t mat) {
    for (int y = 0; y < mat.size1(); y++) {
        for (int x = 0; x < mat.size2(); x++) {
            std::cout << std::setprecision(2) << mat(y,x) << "\t";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char **argv)
{
    matrix_t graph = readMtxFile(argv[1]);

    vector_t vect = vector_t(graph.size1());
    for (int i = 0; i < graph.size1(); i++)
        vect(i) = rand();

    std::cout << "Processing..." << std::endl;
    
    clock_t begin = clock();

    //matrix_t trans = compute_trans(graph);
    //matrix_t tilde = compute_norm(trans);

    vector_t res = matvect_prod(graph, vect);
    clock_t end = clock();

    double elapsed = double(end-begin) / CLOCKS_PER_SEC;
    std::cout << elapsed << " seconds" << std::endl;
    //printMatrix(tilde);

    return 0;
}
