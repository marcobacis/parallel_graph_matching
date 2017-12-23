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
    matrix_t trans = matrix_t(mat.size1(), mat.size2());

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
