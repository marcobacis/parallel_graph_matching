#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <ctime>
#include <iomanip>
#include "nsd.h"
#include "match.h"


void printMatrix(ublas::matrix<float> mat) {

    std::cout << mat.size1() << " " << mat.size2() << std::endl << std::flush;
    for (int y = 0; y < mat.size1(); y++) {
        for (int x = 0; x < mat.size2(); x++) {
            std::cout << std::setprecision(2) << mat(y,x) << "\t" << std::flush;
        }
        std::cout << std::endl << std::flush;
    }
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    //number of "component" (present in the paper)
    int s = 10;

    //determines the convergence (set to 0.8 in the algorithm)
    float alpha = 0.8;

    int n = 100;

    if (argc != 3) {
        std::cout << "Usage" << std::endl;
        std::cout << "\t" << argv[0] << " graph1_path graph2_path" << std::endl << std::endl;
        return 1;
    }

    std::cout << "Initialization" << std::endl << std::flush;

    sparse_t A = readMtxFile(argv[1]);
    sparse_t A_trans = compute_trans(A);
    sparse_t A_tilde = compute_norm(A_trans);

    sparse_t B = readMtxFile(argv[2]);
    sparse_t B_trans = compute_trans(B);
    sparse_t B_tilde = compute_norm(B_trans);

    //initialize W0 and Z0 vectors
    std::vector<vector_t > Z(s, vector_t(A_tilde.size1()));
    std::vector<vector_t > W(s, vector_t(B_tilde.size1()));

    for(int i = 0; i < s; i++){
        for(int j = 0; j < Z[i].size(); j++) {
            Z[i][j] = float(rand()) / RAND_MAX;
        }
        for(int j = 0; j < W[i].size(); j++) {
            W[i][j] = float(rand()) / RAND_MAX;
        }
    }
    //X computation

    //the auction algorithms wants X in form Na x Nb, where Na < Nb
    // so we swap the matrices in the function call if needed

    //given A and B, compute iterates returns a Nb x Na matrix
    //so we swap if Nb > Na

    std::cout << "Computing similarity matrix..." << std::endl;

    bool swap = A.size1() < B.size1();

    int height = swap ? B.size1() : A.size1();

    int width = swap ? A.size1() : B.size1();

    matrix_t X(height, width, 0);

    for (int i = 0; i < s; i++){
        if (swap)
            X += compute_x_iterate(A_tilde, B_tilde, W[i], Z[i], n, alpha);
        else
            X += compute_x_iterate(B_tilde, A_tilde, Z[i], W[i], n, alpha);
        std::cout << "Iteration " << i+1 << " of " << s << std::endl;
    }

    //printMatrix(X);

    // Auction

    return 0;
}
