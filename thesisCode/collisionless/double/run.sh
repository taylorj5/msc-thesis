#! /bin/bash
###runs program for several  matrix sizes and number of threads per block
###stores output in file of format n(sizeofn)m(sizeofm).txt, eg: n1000m1000.txt

matrixsizes=(32 64 128 256 512)
blocksizes=(32 64 128 256 512 1024)

#runs file for each required block size for n x m matrix
for j in "${matrixsizes[@]}"
do
	for i in "${blocksizes[@]}"
	do
		./main -n $j -m $j -s $i -c
	done
done
