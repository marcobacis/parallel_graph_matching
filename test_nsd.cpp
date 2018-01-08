#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <ctime>
#include <iomanip>
#include "nsd.h"
#include "match.h"

void writeMatrix(ublas::matrix<float> mat, std::string filename) {

    std::ofstream ofs(filename);

    ofs << mat.size1() << " " << mat.size2() << std::endl << std::flush;
    for (unsigned int y = 0; y < mat.size1(); y++) {
        for (unsigned int x = 0; x < mat.size2(); x++) {
          ofs << std::to_string(mat(y,x)) << "\t" << std::flush;
        }
        ofs << std::endl;
    }

    ofs.close();
}

int main(int argc, char **argv)
{
    srand(time(NULL));

    // Global parameters
    int s = 10;
    float alpha = 0.8;
    int n = 10;

    // Initialization phase

    MPI_Init(&argc, &argv);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //Dummy matrices to pass to the broadcast/scatter functions
    sparse_t A_tilde, B_tilde;
    sparse_t A_trans, B_trans;

    //Root node
    if(world_rank == 0) {

        if (argc != 3) {
            std::cout << "Usage" << std::endl;
            std::cout << "\t" << argv[0] << " graph1_path graph2_path" << std::endl << std::endl;
            return 1;
        }

        std::cout << "Initialization" << std::endl << std::flush;

        sparse_t A_read = readMtxFile(argv[1]);
        sparse_t B_read = readMtxFile(argv[2]);

        bool swap = A_read.size1() < B_read.size1();
        if (swap) {
            A_trans = compute_trans(B_read);
            B_trans = compute_trans(A_read);
        } else {
            A_trans = compute_trans(A_read);
            B_trans = compute_trans(B_read);
        }

        A_tilde = compute_norm(A_trans);
        B_tilde = compute_norm(B_trans);
    }

    // ALL workers (root included)

    //Broadcast A
    sparse_t A = broadcast_matrix(0, A_tilde);

    //Scatter B
    sparse_t B = scatter_matrix(0, B_tilde);

    //X computation (along with Z initialization, can be changed)

    int height = B.size1();
    int width = A.size1();

    matrix_t X = ublas::zero_matrix<float>(height,width);

    for(int i = 0; i < s; i++){

        //allocate real vector only if root
        vector_t Z_buff, W_buff;
        if(world_rank == 0){
            Z_buff = vector_t(A.size2(),0);
            W_buff = vector_t(B.size2(),0);
        }

        if (world_rank == 0) {
            float sum = 0;
            for(unsigned int j = 0; j < Z_buff.size(); j++) {
                float val = float(rand()) / RAND_MAX;
                Z_buff(j) = val;
                sum +=val;
            }

            if (sum != 0) Z_buff /= sum;

            sum = 0;
            for(unsigned int j = 0; j < W_buff.size(); j++) {
                float val = j < Z_buff.size() ? Z_buff(j) : float(rand()) / RAND_MAX;
                W_buff(j) = val;
                sum += val;
            }
            if(sum != 0) W_buff /= sum;
        }


        vector_t Z = broadcast_vector(0,Z_buff);
        vector_t W = broadcast_vector(0,W_buff);

        std::cout << "Worker " << world_rank << " Iteration " << i << std::endl;

        X += compute_x_iterate_mpi(A, B, Z, W, n, alpha);

    }

    //printMatrix(X);

    std::cout << "Worker" << world_rank << " starting auction" << std::endl;

    //Auction
    runAuction(X.size2(),X);

    std::cout << "Worker " << world_rank << " finished!" << std::endl;

    // Finalize the MPI environment.
    MPI_Finalize();



    return 0;
}
