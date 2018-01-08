#include "nsd.h"


sparse_t readMtxFile(std::string filename) {
    int height, width, nonzeros;
    int y,x;

    std::ifstream fin(filename);

    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');

    fin >> height >> width >> nonzeros;

    sparse_t mat(height, width, nonzeros);

    for (int i = 0; i < nonzeros; i++) {
        fin >> y >> x;
        mat(y-1,x-1) = 1;
    }

    return mat;
}

/*
Fast transposition for sparse matrices, works by grabbing all the nonzero coords,
sorting them and writing the trans matrix.
It's faster because writing with transposed coordinates is a pain if not sorted,
but we sort them so it's a linear pattern after.
*/
sparse_t compute_trans(sparse_t mat) {
    sparse_t trans = sparse_t(mat.size2(), mat.size1());

    typedef unsigned long long coord;

    std::vector<coord> coords;
    coord theight = mat.size2();

    for(is1_t i1 = mat.begin1(); i1 != mat.end1(); ++i1) {
        for(is2_t i2 = i1.begin(); i2 != i1.end(); ++i2) {
            coord y = i2.index1();
            coord x = i2.index2();
            coords.push_back(x * theight + y);
        }
    }

    std::sort(coords.begin(), coords.end());

    for(unsigned int i = 0; i < coords.size(); i++) {
        coord y = coords[i] / theight;
        coord x = coords[i] % theight;
        trans(y,x) = mat(x,y);
    }

    return trans;
}

float sum_elements(matrix_t mat) {
    float sum = 0;
    for(i1_t i1 = mat.begin1(); i1 != mat.end1(); ++i1) {
        for(i2_t i2 = i1.begin(); i2 != i1.end(); ++i2) {
            sum += *i2;
        }
    }

    return sum;
}

sparse_t compute_norm(sparse_t mat) {

    sparse_t tilde = sparse_t(mat);

    float *sums = new float [tilde.size1()];

    for(unsigned int y = 0; y < tilde.size1(); y++)
        sums[y] = 0;

    for(is1_t i1 = tilde.begin1(); i1 != tilde.end1(); ++i1) {
        for(is2_t i2 = i1.begin(); i2 != i1.end(); ++i2) {
            sums[i2.index1()] += *i2;
        }
    }

    for(is1_t i1 = tilde.begin1(); i1 != tilde.end1(); ++i1) {
        int y = i1.index1();
        if (sums[y] != 0) {
            for(is2_t i2 = i1.begin(); i2 != i1.end(); ++i2) {
                int x = i2.index2();
                tilde(y, x) /= sums[y];
            }
        }
    }

    return tilde;
}

vector_t matvect_prod(matrix_t mat, vector_t vect) {
    vector_t result = vector_t(mat.size1(),0);

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


void printMatrix(ublas::matrix<float> mat) {

    std::cout << mat.size1() << " " << mat.size2() << std::endl << std::flush;
    for (unsigned int y = 0; y < mat.size1(); y++) {
        for (unsigned int x = 0; x < mat.size2(); x++) {
            //std::cout << std::setfill('0') << std::setw(5)
          //<< std::fixed << std::setprecision(5) << mat(y,x) << "\t" << std::flush;
          std::cout << mat(y,x) << "\t" << std::flush;
        }
        std::cout << std::endl << std::flush;
    }
}

matrix_t compute_x_iterate(matrix_t A, matrix_t B, vector_t Z, vector_t W, int n,float alpha) {

    vector_t W_i[n];
    vector_t Z_i[n];

    //Iterate over w,z
    W_i[0] = vector_t(W);
    Z_i[0] = vector_t(Z);

    for (int i = 1; i < n; i++) {
        W_i[i] = matvect_prod(B, W_i[i-1]);
        Z_i[i] = matvect_prod(A, Z_i[i-1]);
    }

    matrix_t X = ublas::zero_matrix<float>(B.size1(), A.size1());
    float alpha_pow = 1;
    for(int i = 0; i < n-1; i++) {
        X = X + alpha_pow * outer_prod(W_i[i], Z_i[i]);
        alpha_pow *= alpha;
    }
    X = (1 - alpha) * X + alpha_pow * outer_prod(W_i[n-1], Z_i[n-1]);
    return X;
}

matrix_t compute_x_iterate_mpi(matrix_t A, matrix_t B, vector_t Z, vector_t W, int n,float alpha) {

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    vector_t W_i[n];
    vector_t Z_i[n];

    //create W_i[0] by using a range of W
    int wsize = ((int)W.size() / world_size);
    int rstart =  wsize * rank;
    int rend = rank == world_size-1 ? (int)W.size() : rstart + wsize;

    vector_t W0(rend-rstart);
    for(int i = rstart; i < rend; i++)
        W0(i-rstart) = W(i);

    //Iterate over w,z
    W_i[0] = W0;
    Z_i[0] = vector_t(Z);


    for (int i = 1; i < n; i++) {
        vector_t W_gathered;

        if (i == 1) W_gathered = W;
        else W_gathered = allgather_vector(W_i[i-1]);

        W_i[i] = matvect_prod(B, W_gathered);
        Z_i[i] = matvect_prod(A, Z_i[i-1]);
    }

    matrix_t X = ublas::zero_matrix<float>(B.size1(), A.size1());
    float alpha_pow = 1;
    for(int i = 0; i < n-1; i++) {
        X = X + alpha_pow * outer_prod(W_i[i], Z_i[i]);
        alpha_pow *= alpha;
    }
    X = (1 - alpha) * X + alpha_pow * outer_prod(W_i[n-1], Z_i[n-1]);
    return X;
}

void decompose_matrix(sparse_t mat, int components, std::vector<int> &xs, std::vector<int> &ys, std::vector<float> &vals, int nnz[], int sizes[]) {

    int rowproc = mat.size1() / components;
    int heightmod = mat.size1() % components;

    for(int i = 0; i < components; i++){
        nnz[i] = 0;
        sizes[i * 2] = i == components-1 ? rowproc + heightmod : rowproc;
        sizes[i * 2 + 1] = mat.size2();
    }


    for(is1_t i1 = mat.begin1(); i1 != mat.end1(); ++i1) {
        for(is2_t i2 = i1.begin(); i2 != i1.end(); ++i2) {
            int yidx = i2.index1();
            int xidx = i2.index2();
            int idx = std::min(yidx / rowproc, components-1);
            if (*i2 != 0) {
                xs.push_back(xidx);
                //changes y index to start the submatrix from 0
                ys.push_back(yidx - idx * rowproc);
                vals.push_back(*i2);
                nnz[idx]++;
            }
        }
    }

}

sparse_t compose_matrix(int height, int width, int nnz, int xcoord[], int ycoord[], float vals[]) {
    sparse_t mat(height, width, 0);

    for(int i = 0; i < nnz; i++)
        mat(ycoord[i], xcoord[i]) = vals[i];

    return mat;
}

sparse_t scatter_matrix(int root, sparse_t mat) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int nprocs = 1;
    if (rank == root) // avoids allocating a big "sizes" and "nnz" array to feed scatter/v
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int nnz[nprocs];
    int sizes[nprocs * 2];

    std::vector<int> xs;
    std::vector<int> ys;
    std::vector<float> vals;

    if (rank == root){
        decompose_matrix(mat, nprocs, xs, ys, vals, nnz, sizes);
        std::cout << std::endl;
    }

    //scatter and get sizes and number of nonzeros (used later)
    int size[2];
    int nonzeros;
    MPI_Scatter(sizes, 2, MPI_INT, size, 2, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Scatter(nnz, 1, MPI_INT, &nonzeros, 1, MPI_INT, root, MPI_COMM_WORLD);

    //allocates arrays to recreate matrix
    int *xcoords = new int[nonzeros];
    int *ycoords = new int[nonzeros];
    float *matvals = new float[nonzeros];

    //scatter for root
    if (rank == root) {

        //compute displacements
        int displs[nprocs];
        int sum = 0;
        for (int i = 0; i < nprocs; i++) {
            displs[i] = sum;
            sum += nnz[i];
        }

        MPI_Scatterv(xs.data(), nnz, displs, MPI_INT, xcoords, nonzeros, MPI_INT, root, MPI_COMM_WORLD);
        MPI_Scatterv(ys.data(), nnz, displs, MPI_INT, ycoords, nonzeros, MPI_INT, root, MPI_COMM_WORLD);
        MPI_Scatterv(vals.data(), nnz, displs, MPI_FLOAT, matvals, nonzeros, MPI_FLOAT, root, MPI_COMM_WORLD);
    } else { // for other nodes
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, xcoords, nonzeros, MPI_INT, root, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, ycoords, nonzeros, MPI_INT, root, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, matvals, nonzeros, MPI_FLOAT, root, MPI_COMM_WORLD);
    }

    sparse_t composed = compose_matrix(size[0], size[1], nonzeros, xcoords, ycoords, matvals);

    delete[] xcoords;
    delete[] ycoords;
    delete[] matvals;

    return composed;

}

sparse_t broadcast_matrix(int root, sparse_t mat) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> xs;
    std::vector<int> ys;
    std::vector<float> vals;
    int nnz[1];
    int size[2];

    if (rank ==  root)
        decompose_matrix(mat, 1, xs, ys, vals, nnz, size);

    MPI_Bcast(size, 2, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(nnz, 1, MPI_INT, root, MPI_COMM_WORLD);

    int *xcoords = new int[nnz[0]];
    int *ycoords = new int[nnz[0]];
    float *matvals = new float[nnz[0]];

    if (rank == root){
        for(int i = 0; i < nnz[0]; i++){
            xcoords[i] = xs[i];
            ycoords[i] = ys[i];
            matvals[i] = vals[i];
        }
    }

    MPI_Bcast(xcoords, nnz[0], MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(ycoords, nnz[0], MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(matvals, nnz[0], MPI_FLOAT, root, MPI_COMM_WORLD);

    sparse_t composed = compose_matrix(size[0], size[1], nnz[0], xcoords, ycoords, matvals);

    delete[] xcoords;
    delete[] ycoords;
    delete[] matvals;

    return composed;
}

vector_t broadcast_vector(int root, vector_t vect) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size;

    float *vect_data;

    if (rank == root) {
        size= vect.size();

        MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);

        vect_data = new float[size];

        for(int i = 0; i < size; i++)
            vect_data[i] = vect(i);

    } else {
        MPI_Bcast(&size, 1, MPI_INT, root, MPI_COMM_WORLD);

        vect_data = new float[size];
    }

    MPI_Bcast(vect_data, size, MPI_FLOAT, root, MPI_COMM_WORLD);

    vector_t to_return = vector_t(size,0);

    for(int i = 0; i < size; i++)
        to_return(i) = vect_data[i];

    delete[] vect_data;

    return to_return;

}

vector_t allgather_vector(vector_t local) {
    //sends vector to everyone

    int size = local.size();

    int nprocs = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int sizes[nprocs];

    //gather all the sizes
    MPI_Allgather(&size, 1, MPI_INT, sizes, 1, MPI_INT, MPI_COMM_WORLD);

    //computes displacements
    int tot_size = 0;
    int displs[nprocs];
    for(int i = 0; i < nprocs; i++) {
        displs[i] = tot_size;
        tot_size += sizes[i];
    }

    float *local_buffer = new float[size];
    float *gathered = new float[tot_size];

    for(int l = 0; l < size; l++)
        local_buffer[l] = local(l);

    //gathers the array
    MPI_Allgatherv(local_buffer, size, MPI_FLOAT, gathered, sizes, displs, MPI_FLOAT, MPI_COMM_WORLD);

    delete[] local_buffer;

    vector_t gathered_vect(tot_size, 0);

    for(int i = 0; i < tot_size; i++)
        gathered_vect(i) = gathered[i];

    delete[] gathered;

    return gathered_vect;
}
