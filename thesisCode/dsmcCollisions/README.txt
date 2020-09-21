This directory contains various implementations of DSMC collisions, each stored in its own subdirectory. All implementations are outlined in the writeup, and are briefly explained below. Please refer to the txt file in the parent directory for instructions on loading the required modules and compiling the code. The program contained in the cellTrackerAdvanced subdirectory is the optimal implementation of the code and (unless otherwise stated) all results are obtained using it.

#########DIRECTORY STRUCTURE
naive: this implementation directly computes the collisions between all particles in the simulation, and is very expensive.

cellTrackerGlobal: uses the ''cellTracker'' array to keep track of particle locations in the grid and simplify the collision process.

cellTrackerSurface2d: very similar to the 1d case but uses 2d surface object and can simulate larger problem sizes as a result.

cellTrackerAdvanced: significant improvement over the previous cellTracker implementations, using 2d surface memory to store the cellTracker array, but updates it rather than rerwriting it after each iteration. This is the optimal implementation of the simulation.

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
