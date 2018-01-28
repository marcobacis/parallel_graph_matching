#!/usr/bin/env bash

#mpi=$1
omp=$1

out=results/yeast-yeast_sub500.res.5

for i in `seq $omp -1 2`;
    do
    export OMP_NUM_THREADS=$i
    echo "omp : " $i 

    maxmpi=$[24/$i]
 
    for j in `seq $maxmpi -1 1`;
        do
            echo "mpi : " $j
        
            mpirun -np $j -bind-to socket -map-by socket build/test_nsd graphs/yeast/yeast_sorted.mtx graphs/yeast_subgraph500.mtx  >> $out
        done
    done

export OMP_NUM_THREADS=1

maxmpi=24

for j in `seq $maxmpi -1 13`;
    do
        echo "mpi : " $j
        
        mpirun -np $j build/test_nsd  graphs/yeast/yeast_sorted.mtx graphs/yeast_subgraph500.mtx >> $out
    done


for j in `seq 12 -1 1`;
    do
        echo "mpi : " $j
        
        mpirun -np $j -bind-to socket -map-by socket build/test_nsd  graphs/yeast/yeast_sorted.mtx graphs/yeast_subgraph500.mtx >> $out
    done


#-bind-to socket -map-by socket
# ../graphs/yeast/yeast_sorted.mtx ../graphs/yeast_subgraph500.mtx
# ../graphs/yeast/yeast_sorted.mtx ../graphs/yeast_subgraph1000.mtx
