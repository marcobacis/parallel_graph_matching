#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <ctime>
#include <iomanip>
#include "nsd.h"


/*void worker_process(int rank, int s, int n, float alpha, int nprocs) {
    std::cout << "Im the worker number " << rank << std::endl << std::flush;

    matrix_t A = receive_matrix();

    std::cout << rank << " :A received " << std::endl;

    std::vector<vector_t > Z;
    std::vector<vector_t > W;

    for (int i = 0; i < s; i++)
        Z.push_back(receive_vector(0));

    std::cout << rank << " : Z received (" << Z.size() << " elements)" << std::endl;

    for (int i = 0; i < s; i++)
        W.push_back(receive_vector(0));

    std::cout << rank << " : W received (" << W.size() << " elements)" << std::endl;

    matrix_t B = receive_matrix();

    std::cout << rank << " : B received " << std::endl;


    std::cout << "Process " << rank << "sizes:" << std::endl
    << "A: " << A.size1() << " x " << A.size2() << std::endl
    << "B: " << B.size1() << " x " << B.size2() << std::endl
    << "W: " << W.size() << " vectors of size " << W[0].size() << std::endl
    << "Z: " << Z.size() << " vectors of size " << Z[0].size() << std::endl;

    matrix_t X = ublas::zero_matrix<float>(B.size1(), A.size1());
    for (int i = 0; i < s; i++)
        X += compute_x_iterate_mpi(B, A, W[i], Z[i], n, alpha, nprocs, rank);

    std::cout << rank << " : X computed " << std::endl;

    printMatrix(X);
}


void send_stuff(matrix_t A, matrix_t B, std::vector<vector_t > Z, std::vector<vector_t > W, int nprocs) {

    broadcast_matrix(A, nprocs);

    int s = Z.size();

    for (int i = 0; i < s; i++)
        broadcast_vector(Z[i], nprocs, 0);

    for (int i = 0; i < s; i++)
        broadcast_vector(W[i], nprocs, 0);

    scatter_matrix(B, nprocs);
}
*/

int main(int argc, char **argv)
{
    srand(time(NULL));

    // Global parameters
    int s = 10;
    float alpha = 0.8;
    int n = 100;

    // Initialization phase

    int procs; //number of processes in which the computation is divided (row-wise)

    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    procs = world_size;

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //Root node
    if(world_rank == 0) {

        srand(time(NULL));


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
            float sum = 0;
            for(unsigned int j = 0; j < Z[i].size(); j++) {
                float val = float(rand()) / RAND_MAX;
                Z[i](j) = val;
                sum +=val;
            }

            Z[i] /= sum;

            sum = 0;
            for(unsigned int j = 0; j < W[i].size(); j++) {
                float val = j < Z[0].size() ? Z[i](j) : float(rand()) / RAND_MAX;
                W[i](j) = val;
                sum += val;
            }
            W[i] /= sum;
        }

        //sends the initialization stuff (Atilde, Btilde, W, Z) around
        bool swap = A.size1() < B.size1();

        /*if (swap)
            send_stuff(B_tilde, A_tilde, W, Z, procs-1);
        else
            send_stuff(A_tilde, B_tilde, Z, W, procs-1);*/

        matrix_t A_recv = scatter_matrix(0, A_tilde);
        std::cout << "Worker " << world_rank << " received a "
         << A_recv.size1() << " by " << A_recv.size2() << " matrix" << std::endl << std::flush;
    }

    // Other worker nodes
    else{
        matrix_t dummy;
        matrix_t A_recv = scatter_matrix(0, dummy);

        std::cout << "Worker " << world_rank << " received a "
         << A_recv.size1() << " by " << A_recv.size2() << " matrix" << std::endl << std::flush;

        //worker_process(world_rank, s, n, alpha, procs-1);
    }

    // Finalize the MPI environment.
    MPI_Finalize();



    return 0;
}
