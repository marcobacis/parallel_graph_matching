#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include "nsd.h"

std::vector<int> runAuction(int nb, matrix_t X); // na <= nb , na buyers, nb objects
std::vector<int>  auctionSerial(matrix_t X);

std::vector<int> collectMatch(std::vector<int> match, int nLocal);
