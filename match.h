#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include "nsd.h"

void runAuction(int nb, matrix_t X); // na <= nb , na buyers, nb objects
void auctionSerial(matrix_t X);

void collectMatch(std::vector<int> match, int nLocal);
