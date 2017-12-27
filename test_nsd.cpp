#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <ctime>
#include <iomanip>
#include "nsd.h"


void printMatrix(ublas::matrix<float> mat) {

    std::cout << mat.size1() << " " << mat.size2() << std::endl << std::flush;
    for (int y = 0; y < mat.size1(); y++) {
        for (int x = 0; x < mat.size2(); x++) {
            std::cout << std::setprecision(2) << mat(y,x) << "\t" << std::flush;
        }
        std::cout << std::endl << std::flush;
    }
}


void worker_process(int rank) {
    std::cout << "Im the worker number " << rank << std::endl << std::flush;

    matrix_t mat = receive_matrix();

    std::cout << "Worker " << rank << " received a " << mat.size1() << "x" << mat.size2() << " matrix" << std::endl;
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    int procs; //number of processes in which the computation is divided (row-wise)

    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    procs = world_size; //doesn't count the root

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //Root node
    if(world_rank == 0) {

        std::cout << "Initialization" << std::endl << std::flush;

        sparse_t A = readMtxFile(argv[1]);
        sparse_t A_trans = compute_trans(A);
        sparse_t A_tilde = compute_norm(A_trans);

        sparse_t B = readMtxFile(argv[1]);
        sparse_t B_trans = compute_trans(B);
        sparse_t B_tilde = compute_norm(B_trans);

        vector_t Z(A_tilde.size2());
        for(int i = 0; i < Z.size(); i++) {
            Z(i) = float(rand()) / RAND_MAX;
        }
        vector_t W(B_tilde.size2());
        for(int i = 0; i < W.size(); i++) {
            W(i) = float(rand()) / RAND_MAX;
        }

        std::cout << "Sending to " << procs << " workers" << std::endl << std::flush;

        //printMatrix(A_tilde);

        scatter_matrix(A_tilde,procs);

        worker_process(world_rank);

    }

    // Other worker nodes
    else{
        worker_process(world_rank);
    }

    // Finalize the MPI environment.
    MPI_Finalize();



    return 0;
}
