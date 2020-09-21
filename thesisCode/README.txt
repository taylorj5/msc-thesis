#########INTRODUCTION
The code in this dircetory is part of my submission for the MSc in HPC. While there are many further subdirectories containing versions of the code at different stages of development, the main code for the thesis is contained in the dsmcCollisions subdirectory. The most optimised version of the code is contained in subdirectory dsmcCollisions/cellTrackeradvanced, and this is used for most of the results in the project, unless otherwise stated in the writeup. The other subdirectories in the dsmcCollisions directory are also used (or at least mentioned) in the writeup, though the majority of the program remains the same, with the functions pertaining to the collision process being varied.

#########COMPILING CODE
Each folder contains a single version of the serial and GPU code. When inside a directory, 
the correct modules must be loaded by calling

	source ./init.sh

Then the code can be built using the simple command

	make

or 

	make -j 4 

to build the target more quickly in parallel using the 4 cores on cuda-01. Information on running the code is contained in each subdirectory.


#########DIRECTORY STRUCTURE
This directory contains three subdirectories, each relating to a different approach to the simulation. They are briefly described below, and a more detailed description can be found in the writeup.  The project primarily focuses on the DSMCCollisions, as it is the most complex and therefore the most interesting problem. The other implementations are precursors to the DSMC collisions, and were used mainly to develop an understanding of the problem.

- collisionless: this code solves the simple case where particle collisions are neglected

- mccCollisions: this solves the case where collisions are included and approximated using a Monte-Carlo collision with a ''cloud''

- dsmcCollisions: this solves the case where the collisions are directly-simulated for particles in each cell. The bulk of the thesis is based on parallelising this code.
