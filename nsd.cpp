#include "nsd.h"


matrix_t readMtxFile(std::string filename) {
    int height, width, nonzeros;
    int y,x;

    std::ifstream fin(filename);

    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');

    fin >> height >> width >> nonzeros;

    ublas::mapped_matrix<float> mat(height, width, nonzeros);

    for (int i = 0; i < nonzeros; i++) {
        fin >> y >> x;
        mat(y-1,x-1) = 1;
    }

    return mat;
}

matrix_t compute_trans(matrix_t mat) {
    matrix_t trans = matrix_t(mat.size2(), mat.size1());

    for(i1_t i1 = mat.begin1(); i1 != mat.end1(); ++i1) {
        for(i2_t i2 = i1.begin(); i2 != i1.end(); ++i2) {
            int y = i2.index1();
            int x = i2.index2();
            trans(x,y) = *i2;
        }
    }

    return trans;
}

matrix_t compute_norm(matrix_t mat) {

    matrix_t tilde = matrix_t(mat);

    float *sums = new float [tilde.size1()];

    for(i1_t i1 = tilde.begin1(); i1 != tilde.end1(); ++i1) {
        for(i2_t i2 = i1.begin(); i2 != i1.end(); ++i2) {
            sums[i2.index1()] += *i2;
        }
    }

    for(i1_t i1 = tilde.begin1(); i1 != tilde.end1(); ++i1) {
        for(i2_t i2 = i1.begin(); i2 != i1.end(); ++i2) {
            int y = i2.index1();
            int x = i2.index2();
            tilde(y, x) /= sums[y];
        }
    }

    return tilde;
}

vector_t matvect_prod(matrix_t mat, vector_t vect) {
    vector_t result = vector_t(mat.size1());

    for(i1_t i1 = mat.begin1(); i1 != mat.end1(); ++i1) {
        float sum = 0;
        int y = i1.index1();
        for(i2_t i2 = i1.begin(); i2 != i1.end(); ++i2) {
            int x = i2.index2();
            sum += *i2 * vect(x);
        }
        result(y) = sum;
    }
    return result;
}

matrix_t compute_x_iterate(matrix_t A, matrix_t B, vector_t Z, vector_t W, int s, int n,float alpha) {

    vector_t W_i[n];
    vector_t Z_i[n];

    //Iterate over w,z
    W_i[0] = vector_t(W);
    Z_i[0] = vector_t(Z);

    for (int i = 1; i < n; i++) {
        W_i[i] = matvect_prod(B, W_i[i-1]);
        Z_i[i] = matvect_prod(A, Z_i[i-1]);
    }

    matrix_t X(A.size2(), B.size2());
    float alpha_pow = 1;
    for(int i = 0; i < n-1; i++) {
        X += alpha_pow * outer_prod(W_i[i], Z_i[i]);
        alpha_pow *= alpha;
    }
    X = (1 - alpha) * X + alpha_pow * outer_prod(W_i[n-1], Z_i[n-1]);

    return X;
}

void decompose_matrix(matrix_t mat, int components, std::vector<int> xs[], std::vector<int> ys[], std::vector<float> vals[], int nnz[]) {

    for(int i = 0; i < components; i++) nnz[i] = 0;

    int rowproc = mat.size1() / components;

    for(i1_t i1 = mat.begin1(); i1 != mat.end1(); ++i1) {
        for(i2_t i2 = i1.begin(); i2 != i1.end(); ++i2) {
            int yidx = i2.index1();
            int xidx = i2.index2();
            int idx = yidx / rowproc;

            xs[idx].push_back(xidx);
            ys[idx].push_back(yidx);
            vals[idx].push_back(*i2);
            nnz[idx]++;
        }
    }

}

void scatter_matrix(matrix_t mat, int nproc) {
    std::vector<int> xs[nproc];
    std::vector<int> ys[nproc];
    std::vector<float> vals[nproc];
    int nnz[nproc];

    decompose_matrix(mat, nproc, xs, ys, vals, nnz);

    for(int i = 0; i < nproc; i++){
        int dest = i;
        int sizes[3] = {mat.size1()/nproc, mat.size2(), nnz[i]};
        MPI_Send(sizes, 3, MPI_INT, dest, MSG_MATRIX_SIZE, MPI_COMM_WORLD);
        MPI_Send(&xs[i][0], nnz[i], MPI_INT, dest, MSG_MATRIX_X, MPI_COMM_WORLD);
        MPI_Send(&ys[i][0], nnz[i], MPI_INT, dest, MSG_MATRIX_Y, MPI_COMM_WORLD);
        MPI_Send(&vals[i][0], nnz[i], MPI_FLOAT, dest, MSG_MATRIX_VALS, MPI_COMM_WORLD);
    }
}

void broadcast_matrix(matrix_t mat, int nproc) {

    std::vector<int> xs[1];
    std::vector<int> ys[1];
    std::vector<float> vals[1];
    int nnz[1];

    decompose_matrix(mat, 1, xs, ys, vals, nnz);

    for(int i = 0; i < nproc; i++){
        int dest = i;
        int sizes[3] = {mat.size1(), mat.size2(), nnz[0]};
        MPI_Send(sizes, 3, MPI_INT, dest, MSG_MATRIX_SIZE, MPI_COMM_WORLD);
        MPI_Send(&xs[0][0], nnz[i], MPI_INT, dest, MSG_MATRIX_X, MPI_COMM_WORLD);
        MPI_Send(&ys[0][0], nnz[i], MPI_INT, dest, MSG_MATRIX_Y, MPI_COMM_WORLD);
        MPI_Send(&vals[0][0], nnz[i], MPI_FLOAT, dest, MSG_MATRIX_VALS, MPI_COMM_WORLD);
    }

}


matrix_t receive_matrix() {
    MPI_Status status;
    int from = 0;
    //First, receives matrix size
    int sizes[3];
    MPI_Recv(sizes, 3, MPI_INT, from, MSG_MATRIX_SIZE, MPI_COMM_WORLD, &status);

    int height = sizes[0];
    int width = sizes[1];
    int nnz = sizes[2];

    matrix_t mat(height, width);


    int y[nnz], x[nnz];
    float vals[nnz];

    MPI_Recv(x, nnz, MPI_INT, from, MSG_MATRIX_X, MPI_COMM_WORLD, &status);
    MPI_Recv(y, nnz, MPI_INT, from, MSG_MATRIX_Y, MPI_COMM_WORLD, &status);
    MPI_Recv(vals, nnz, MPI_FLOAT, from, MSG_MATRIX_VALS, MPI_COMM_WORLD, &status);

    for(int i = 0; i < nnz; i++) {
        mat(y[i] % height, x[i] % width) = vals[i];
    }

    return mat;
}
