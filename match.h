/*

Functions to implement

- run_auction
- auction_iteration(M, I, prices, X)
    - update_epsilon(n, theta, sparse)

*/

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include "nsd.h"

void auction(int na, int nb, float* X);

int auctionSerial(matrix_t X);
