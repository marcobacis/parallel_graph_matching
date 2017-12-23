/*

Functions to implement

- run_auction
- auction_iteration(M, I, prices, X)
    - update_epsilon(n, theta, sparse)

*/

#include <mpi.h>
#include <omp.h>
#include <stdio.h>

void auction(int na, int nb, int** X);
