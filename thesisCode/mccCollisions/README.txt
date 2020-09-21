This directory contains code to simulate a collisional plasma using MCC collisions (in double precision)


#########RUNNING THE CODE
When the code is compiled as outlined in the parent directory as follows:

- source init.sh
- make clean; make -j 4

The code can then be run for a problem of size xDim * yDim with a block size of blockSize as below:

- ./main -n xDim -m yDim -s blockSize

Other flags that can be used include:
'-c' to skip the serial computation
'-g' to skip the gpu computation
'-x'/'-y' to set the size of the simulation space
'-l' to set the grid size in both dimensions
'-t' to set the tmax variable (to determine limit of simulation)
'-f' to set up a file for output (this preps an output file in case it is desired though there is no explicit code to output results to file as it was not used)
