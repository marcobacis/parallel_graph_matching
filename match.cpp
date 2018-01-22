#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <vector>
#include <unordered_set>
#include "match.h"
#include "nsd.h"

using namespace std;

float n=10e6;

int worldSize, worldRank;

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

vector<int> auction(int na, int nb, matrix_t X){ // na <= nb , na buyers, nb objects
    int nLocal = X.size1();

    vector <int> match(nLocal,-1);
    vector <int> assigned(nb,-1);

    unordered_set <int> freeBuyer;
    for (int i=0; i<nLocal; i++)
        freeBuyer.insert(i);

    unordered_set <int> freeBuyerGlobal;
    for (int i=0; i<na; i++)
        freeBuyerGlobal.insert(i);

    vector <float> price(nb,0);

    float localPrice;
    float sendPrice[2];
    float *receivedPrices = (float *)malloc(sizeof(float) * (worldSize*2));

    int *recvBuyerLenghts = (int *)malloc(sizeof(int) * (worldSize));
    vector <int> recvBuyer;
    vector <int> sendBuyer;

    /* Initialize epsilon */
    float teta = 16;
    float xi = 2;
    float epsilon = (n+1)/teta;
    float deps = 1/(n+1);
    float gamma = (n+1)/teta;
    unsigned int delta = floor(min(na/xi, n/teta));
    float min_gamma = 5e-10;

    bool debug = false;

    while (!freeBuyerGlobal.empty()) {
        epsilon = teta/(n+1);

		gamma = (n+1)/teta;

        while (freeBuyerGlobal.size() > delta) {

            /* Bidding */

            if (debug) {
            cout << worldRank << " : price ";
            for (int i=0;i<nb;i++)
                cout << price[i] << " ";
            cout << "\n";

            cout << worldRank << " : sendBuyer ";
            for (unsigned int i = 0; i < sendBuyer.size(); i++) {
                auto it = sendBuyer.begin();
                advance(it,i);
                cout << *it << " ";
            }
            cout << "\n";

            cout << worldRank << " : globalB ";
            for (unsigned int i = 0; i < freeBuyerGlobal.size(); i++) {
                auto it = freeBuyerGlobal.begin();
                advance(it,i);
                cout << *it << " ";
            }
            cout << "\n";

            cout << worldRank << " : localB ";
            for (unsigned int i = 0; i < freeBuyer.size(); i++) {
                auto it = freeBuyer.begin();
                advance(it,i);
                cout << *it << " ";
            }
            cout << "\n";
            }

            struct BidResult res = {.obj=-1, .buyer=-1, .maxP=-FLT_MAX, .secondP=-FLT_MAX};

            #pragma omp parallel
            {
            #pragma omp for collapse(2) reduction(bidReduce:res)
            for (unsigned int i = 0; i < freeBuyer.size(); i++) {
                for (int j=0; j<nb; j++) {
                    auto it = freeBuyer.begin();
                    advance(it,i);
                    if (X(*it,j) - price[j] > res.maxP) {
                        if (j != res.obj) {
                            res.secondP = res.maxP;
                        }
                        res.obj = j;
                        res.maxP = X(*it,j) - price[j];
                        res.buyer = *it;
                    } else if (j!=res.obj && X(*it,j) - price[j] > res.secondP) {
                        res.secondP = X(*it,j) - price[j];
                    }
                }
            }
            }

            if (res.secondP == -FLT_MAX)
                res.secondP = 0;

            /* Price update */
            localPrice = price[res.obj] + (res.maxP - res.secondP + epsilon * ((rand()%100)/50));

            if ((debug || true) && res.obj!=-1 && worldRank==0)
                cout << worldRank <<  " : localbuyer " << res.buyer << " make offer for " << res.obj << " offering " << localPrice \
                << " # X value : " << X(res.buyer,res.obj) << " old price : " << price[res.obj] << " maxP : " << res.maxP \
                << " secondP : " << res.secondP << " eps : " << epsilon << " delta " << delta << " gamma " << gamma << "\n";

            /* Gather changed prices */
            sendBuyer.clear();
            sendPrice[0] = localPrice; //price
            sendPrice[1] = res.obj; // object
            MPI_Allgather(sendPrice,2,MPI_FLOAT,receivedPrices,2,MPI_FLOAT,MPI_COMM_WORLD);
            for (int i=0; i<worldSize*2; i+=2) {
                if (price[receivedPrices[i+1]] < receivedPrices[i]) {
                    price[receivedPrices[i+1]] = receivedPrices[i];
                    //se qualcuno dei miei local aveva comprato quell'oggetto ora lo perde.
                    if (assigned[receivedPrices[i+1]] != -1) {

                        if (debug)
                            cout << worldRank <<  " : localbuyer " << assigned[receivedPrices[i+1]] << " lose obj " << receivedPrices[i+1] << "\n";

                        sendBuyer.push_back(assigned[receivedPrices[i+1]] + (worldRank*(na/worldSize)) + 1);

                        match[assigned[receivedPrices[i+1]]] = -1;
                        freeBuyer.insert(assigned[receivedPrices[i+1]]);
                        assigned[receivedPrices[i+1]] = -1;


                    }
                } else if (price[receivedPrices[i+1]] == receivedPrices[i] && receivedPrices[i+1] == res.obj) {
                    //se qualcuno ha offerto lo stesso prezzo del mio vincitore local e il suo indice globale è minore del mio, perdo l'oggetto
                    localPrice--;
				}
            }

            /* Check winner */
            if (localPrice == price[res.obj]) {
                match[res.buyer] = res.obj;
                assigned[res.obj] = res.buyer;
                freeBuyer.erase(res.buyer);

                if (debug)
                    cout << worldRank <<  " : localbuyer " << res.buyer << " get obj " << res.obj << "\n";

                sendBuyer.push_back(-(res.buyer + (worldRank*(na/worldSize)) + 1));
            }

            /* Global freebuyer update */
            // calcolo lunghezza totale
            int localLenght = sendBuyer.size();
            MPI_Allgather(&localLenght,1,MPI_INT,recvBuyerLenghts,1,MPI_INT,MPI_COMM_WORLD);
            unsigned int globalLenght=recvBuyerLenghts[0];
            int displ[worldSize];
            displ[0]=0;
            for (int i=1;i<worldSize;i++){
                globalLenght += recvBuyerLenghts[i];
                displ[i] = displ[i-1] + recvBuyerLenghts[i-1];
            }

            //colleziono singoli effettivi update
            if (recvBuyer.size() < globalLenght)
                recvBuyer.resize(globalLenght);

            MPI_Allgatherv(&sendBuyer.front(),localLenght,MPI_INT,&recvBuyer.front(),recvBuyerLenghts,displ,MPI_INT,MPI_COMM_WORLD);

            for (unsigned int i=0;i<globalLenght;i++){
                if (recvBuyer[i]>0) {
                    freeBuyerGlobal.insert(recvBuyer[i]-1);
                } else if (recvBuyer[i]<0) {
                    freeBuyerGlobal.erase((-recvBuyer[i])-1);
                }
            }

            if (worldRank==0) cout << na - freeBuyerGlobal.size() << " / " << na << " node matched\n";

            if (worldRank==0 && debug) {
				int del=0;
				int add=0;
				for (unsigned int i=0;i<globalLenght;i++){
        	        if (recvBuyer[i]>0) {
						cout << -(recvBuyer[i]-1) << " ";
						del++;
        	        } else if (recvBuyer[i]<0) {
						cout << (-recvBuyer[i])-1 << " ";
        	            add++;
        	        }
            	}
				cout << "\nAssigned " << add << " ; Losed " << del << "\n";
			}

            /* update epsilon */
            if (gamma > epsilon) {
                epsilon *= xi;
            } else {
                epsilon = gamma;
            }
            if(gamma > min_gamma && rand()%24==0)
                gamma /= xi;
        }
        delta /= xi;
        teta *= xi;
		cout << "Exit nested while ######## " << delta << " " << teta << "\n";
		//epsilon = max(deps,epsilon-deps);
    }

    if (debug){
        cout << worldRank << " : Result : \n";
        for (int i=0;i<nLocal;i++)
            cout << worldRank << " : " << i + (worldRank*(na/worldSize)) << " <-> " << match[i] << "\n";
    }

    /* Gather all results */
    vector<int> res = collectMatch(match,nLocal);

    return res;

}

vector<int> collectMatch(vector<int> match, int nLocal){
    vector<int> allMatch;

    int disps[worldSize];
    int contv[worldSize];

    if (worldRank == worldSize-1){
        MPI_Send(&nLocal,1,MPI_INT,0,0,MPI_COMM_WORLD);
    }

    if (worldRank==0) {
        int lastSize;
        MPI_Recv(&lastSize,1,MPI_INT,worldSize-1,0,MPI_COMM_WORLD,NULL);
        allMatch.resize(nLocal*(worldSize-1) + lastSize);

        disps[0]=0;
        contv[0]=nLocal;
        for (int i=1;i<worldSize;i++){
            disps[i] = disps[i-1] + nLocal;
            contv[i] = nLocal;
        }
        contv[worldSize-1] = lastSize;
    }


    MPI_Gatherv(&match.front(),nLocal,MPI_INT,&allMatch.front(),contv,disps,MPI_INT,0,MPI_COMM_WORLD);

    if (worldRank==0){
        for (unsigned int i=0;i<allMatch.size();i++)
            cout << i << " <-> " << allMatch[i] << "\n";
    }

    return allMatch;
}

vector<int> runAuction(int nb, matrix_t X){
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    int na;
    int naLocal=X.size1();

    MPI_Allreduce(&naLocal,&na,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

    return auction(na,nb,X);
}

void auctionSerial(matrix_t X){
    /* Must be runned with only 1 MPI process */

    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    auction(X.size1(),X.size2(),X);

    MPI_Finalize();

}
