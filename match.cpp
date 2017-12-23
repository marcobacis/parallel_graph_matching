#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <cfloat>
#include <vector>
#include <unordered_set>
#include "match.h"

using namespace std;

int n=10000; //TODO a cosa è uguale n ? Al numero di iterazioni ?

void auction(int na, int nb, float *x){
    vector <int> match(na,-1);
    vector <int> assigned(nb,-1);
    unordered_set <int> freeBuyer; //TODO or class template specialization<vector>   std::vector<bool>
    for (int i=0; i<na; i++)
        freeBuyer.insert(i);
    vector <float> price(nb,0);

    float (*X)[nb] = (float (*)[nb]) x;

    /* Initialize epsilon */
    int teta = 16;
    int xi = 2;
    float epsilon = (n+1)/teta;
    float deltaeps = 1/(n+1);
    float gamma = (n+1)/teta;

    int bestObj;
    int maxProfit;
    int secondProfit;
    int buyer;

    while (!freeBuyer.empty()) {
        /* Bidding */
        maxProfit = FLT_MIN;
        secondProfit = FLT_MIN;
        for (auto it = freeBuyer.begin(); it != freeBuyer.end(); it++) {
            for (int j=0; j<nb; j++) {
                if (X[*it][j] - price[j] > maxProfit) {
                    bestObj = j;
                    maxProfit = X[*it][j] - price[j];
                    buyer = *it;
                }
            }
        }
        //TODO Si possono unire se si tiene traccia del secondo ogetto
        for (auto it = freeBuyer.begin(); it != freeBuyer.end(); it++) {
            for (int j=0; j<nb; j++) {
                if (j!=bestObj && X[*it][j] - price[j] > secondProfit && X[*it][j] - price[j] < maxProfit) {
                    secondProfit = X[*it][j] - price[j];
                }
            }
        }

        if (secondProfit == FLT_MIN)
            secondProfit = 0;

        /* Price update */
        price[bestObj] += (maxProfit - secondProfit + epsilon);

        /* Assignment */

        /* delete previous match (if exists)
           and insert the old owner in free buyer */
        if (assigned[bestObj] != -1 && assigned[bestObj]!=buyer) {
            match[assigned[bestObj]] = -1;
            freeBuyer.insert(assigned[bestObj]);
        }

        /* make the new match */
        match[buyer] = bestObj;
        assigned[bestObj] = buyer;
        freeBuyer.erase(buyer);

        /* update epsilon */
        epsilon = max(epsilon,epsilon-deltaeps);
    }

    for (int i=0;i<na;i++)
        cout << i << " <-> " << match[i] << "\n";
}

int main(int argc, char** argv) {
    float x[] = {1,2,3,4,5,6,7,8,9};
    auction(3,3,x);

    return 0;
}
