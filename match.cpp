#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <cfloat>
#include <vector>
#include <unordered_set>
#include "match.h"

using namespace std;

int n=10; //TODO a cosa è uguale n ? Al numero di iterazioni ?

struct BidResult {int obj; int buyer; float maxP; float secondP;};

BidResult cmp(BidResult reduced, BidResult current) {
    BidResult x;
    if (current.maxP >= reduced.maxP) {
        x.maxP = current.maxP;
        x.obj = current.obj;
        x.buyer = current.buyer;
        if (reduced.maxP >= current.secondP && current.obj != reduced.obj) {
            x.secondP = reduced.maxP;
        } else {
            x.secondP = current.secondP;
        }
    } else {
        x.maxP = reduced.maxP;
        x.obj = reduced.obj;
        x.buyer = reduced.buyer;
        if (current.maxP >= reduced.secondP && reduced.obj != current.obj) {
            x.secondP = current.maxP;
        } else {
            x.secondP = reduced.secondP;
        }
    }
    return x;
}

#pragma omp declare reduction \
    (bidReduce : struct BidResult : omp_out = cmp(omp_out,omp_in)) \
     initializer(omp_priv= {.obj=-1, .buyer=-1, .maxP=-FLT_MAX, .secondP=-FLT_MAX} )

void auction(int na, int nb, float *x){ // na <= nb , na buyers, nb objects
    vector <int> match(na,-1);
    vector <int> assigned(nb,-1);
    unordered_set <int> freeBuyer; //TODO or class template specialization<vector>   std::vector<bool>
    for (int i=0; i<na; i++)
        freeBuyer.insert(i);
    vector <float> price(nb,0);

    float (*X)[nb] = (float (*)[nb]) x;

    /* Initialize epsilon */
    float teta = 16;
    float xi = 2;
    float epsilon = (n+1)/teta;
    float deps = 1/(n+1);
    float gamma = (n+1)/teta;
    float delta = floor(min(na/xi, n/teta));

    while (!freeBuyer.empty()) {
        epsilon = teta/(n+1);
        while (freeBuyer.size() > delta) {
            /* Bidding */

            struct BidResult res = {.obj=-1, .buyer=-1, .maxP=-FLT_MAX, .secondP=-FLT_MAX};

            #pragma omp parallel num_threads(8)
            {
            #pragma omp for collapse(2) reduction(bidReduce:res)
            for (int i = 0; i < freeBuyer.size(); i++) {
                for (int j=0; j<nb; j++) {
                    auto it = freeBuyer.begin();
                    advance(it,i);
                    if (X[*it][j] - price[j] > res.maxP) {
                        if (j != res.obj) {
                            res.secondP = res.maxP;
                        }
                        res.obj = j;
                        res.maxP = X[*it][j] - price[j];
                        res.buyer = *it;
                    } else if (j!=res.obj && X[*it][j] - price[j] > res.secondP) {
                        res.secondP = X[*it][j] - price[j];
                    }
                }
            }
            }

            if (res.secondP == -FLT_MAX)
                res.secondP = 0;

            /* Price update */
            price[res.obj] += (res.maxP - res.secondP + epsilon);

            /* Assignment */

            /* delete previous match (if exists)
               and insert the old owner in free buyer */
            if (assigned[res.obj] != -1 && assigned[res.obj]!=res.buyer) {
                match[assigned[res.obj]] = -1;
                freeBuyer.insert(assigned[res.obj]);
            }

            /* make the new match */
            match[res.buyer] = res.obj;
            assigned[res.obj] = res.buyer;
            freeBuyer.erase(res.buyer);

            /* update epsilon */
            if (gamma > epsilon) {
                epsilon *= xi;
            } else {
                epsilon = gamma;
            }
            gamma /= xi;
        }
        delta /= xi;
        teta *= xi;
    }

    for (int i=0;i<na;i++)
        cout << i << " <-> " << match[i] << "\n";
}

int main(int argc, char** argv) {
    float x[] = {1,3,3, 4,5,7, 8,8,9};
    auction(3,3,x);

    return 0;
}
