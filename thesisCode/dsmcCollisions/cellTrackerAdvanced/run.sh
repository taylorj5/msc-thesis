#! /bin/bash
###runs program for range of problem sizes and number of threads per block

#matrixsizes=(32 64 128 256 512)
matrixsizes=(32 64 128 256 512 1024 1536)
#blocksizes=(32 64 128 256 512 1024)
blocksizes=(32)

#runs file for each required block size for n x m matrix
for j in "${matrixsizes[@]}"
do
	for i in "${blocksizes[@]}"
	do
		./main -n $j -m $j -s $i -c
		#./main -n $j -m $j -s $i -g
		#nvprof ./main -n $j -m $j -s $i -c
	done
done
