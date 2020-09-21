/**
 * \file main.cc
 * \brief Main function for the simulation code
 * \author J. Taylor
 * \version 1.0
 * \date 2020-06-05
 */

#include <iostream>
#include <vector>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include "params.h"
#include "particle.h"
#include "grid.h"
#include "fourier.h"
#include "pic.h"

//function definitions
void initialiseParticles(std::vector<Particle> &particles, std::shared_ptr<Params> p);
void cellTracking(std::vector<Particle> particles, Grid& grid, std::shared_ptr<Params> p);
void collisions(std::mt19937& gen, std::vector<Particle> &particles, std::shared_ptr<Params> p, Grid& grid);
void checkResults(std::vector<Particle> &particles, double* xGpu, double *yGpu);
void checkError(std::vector<Particle> &particles, double* xGpu, double *yGpu, std::shared_ptr<Params> p);

//cuda wrapper functions
extern double picFloatGpu(double *positionX, double *positionY, double *velocityX, double *velocityY, double *xResult, double *yResult, std::shared_ptr<Params> p);

/**
 * \brief Main function
 */
int main(int argc, char **argv) {
	int i, opt;
	char *filename;				//filename for output
	bool fileFlag = false;		 	//flag for '-f' input
	bool cpu = true, gpu = true;		//False => skip cpu/gpu calculation
	double initTime, timeCpu, timeGpu;
	struct timeval start, end;
	auto p = std::make_shared<Params>();	//creating Params object for simulation

	//taking command line arguments
	while ((opt = getopt(argc, argv, "x:y:l:n:m:t:f:s:cg")) != -1) {
		switch(opt) {
			case 'x':
				p->xDim = atoi(optarg); break; 			//setting size of grid in x-dimension
			case 'y':
				p->yDim = atoi(optarg); break; 			//setting size of grid in y-dimension
			case 'l':
				p->nx = atoi(optarg);
				p->ny = atoi(optarg);  break; 			//setting number of grid points
			case 'n':
				p->numParticlesX = atoi(optarg); break;
			case 'm':
				p->numParticlesY = atoi(optarg); break; 	//setting simulation size (num. particles)
			case 't':
				p->tmax = atof(optarg); break;			//setting max time of simulation
			case 'f':
				filename = optarg; 				//setting filename for output
				fileFlag = true; break;	 		
			case 's':
				p->blockSize = atoi(optarg); break;		//setting block size for device
			case 'c':
				cpu = false; break;
			case 'g':
				gpu = false; break;
			case '?':
				if (isprint(optopt)) {
					fprintf(stderr, "Unknown option %c\n", optopt); break;
				} else {
					fprintf(stderr, "Unknown option character $x\n", optopt); break;
				}
		}
	}

	//setting some parameters based on inputs
	p->dx = (double) p->xDim/p->nx;
	p->dy = (double) p->yDim/p->ny;
	p->area = p->dx * p->dy;
	p->numParticles = p->numParticlesX * p->numParticlesY;

	//setting seed for random number generator
	p->seed = time(NULL);

	//error checking input parameters
	if (p->numParticles < 1 || p->xDim < 0 || p->yDim < 0 || p->nx < 2 || p->ny < 2 || p->vdx < 0.0 || p->vdy < 0.0 || p->vtx < 0.0 || p->vty < 0.0 || p->dt < 0.0 || p->tmax < 0.0) {
		std::cout << "Error: invalid input arguments" << '\n';
		exit(1);
	}

	double t = 0;

	//creating simulation grid and particles for serial implementation
	Grid grid(p->nx, p->ny, p->numParticles, p);	//create instance of class Grid
	std::vector<Particle> particles;		//create vector of Particles

	//creating arrays to copy to device for parallel implementation
	double *positionX= new double[p->numParticles];
	double *positionY= new double[p->numParticles];
	double *velocityX= new double[p->numParticles];
	double *velocityY= new double[p->numParticles];

	//setting up output file
	if (fileFlag == true) {
		std::ofstream myfile;
		myfile.open(filename, std::ios::trunc);
		if (myfile.is_open()) {
			myfile.close();
		}
	}

	/*	SERIAL IMPLEMENTATION	 */
	////////////STEP 1////////////
	gettimeofday(&start, NULL);

	//initialise the particles on the grid
	initialiseParticles(particles, p);

	i=0;
	//copying particle data to containers to send to GPU
	for (std::vector<Particle>::iterator it=particles.begin(); it!=particles.end(); ++it) {
		Point pos = it->getPosition();
		Point vel = it->getVelocity();

		positionX[i] = pos.x;
		positionY[i] = pos.y;
		velocityX[i] = vel.x;
		velocityY[i] = vel.y;
		i++;
	}
	gettimeofday(&end, NULL);
	initTime=((end.tv_sec + end.tv_usec*0.000001) - (start.tv_sec + start.tv_usec*0.000001));

	//serial computation
	if (cpu) {
		std::mt19937 gen{10}; //seed RNG
		timeCpu = 0.0;
		gettimeofday(&start, NULL);
		int count = 0;
		while (t < p->tmax) {
			//distribute charge onto grid
			grid.chargeAssignment(particles);
		
			//compute electrostatic potential from charge distribution
			fftPoissonSolver(grid, p);
	
			//compute electric field from electrostatic potential
			grid.computeElectricField();
	
			//interpolate E to particle position and compute F
			for (std::vector<Particle>::iterator it = particles.begin(); it!=particles.end(); ++it) {
				it->forceCalculation(grid, p);
			}
	
			//solve particle equation of motion
			for (std::vector<Particle>::iterator it = particles.begin(); it!=particles.end(); ++it) {
				it->positionUpdate(p);
			}
	
			cellTracking(particles, grid, p);

			//simulate collisions
			collisions(gen, particles, p, grid);
			
			/*
			//printing cell occupancy every x iterations
			int x = 100;
			if (count%x == 0) {
					std::cout << "\n\nTime t = " << t << '\n';
				for (int i=0; i<p->nx*p->ny; i++) {
					int tmp = grid.cellTracker[i].size();
					std::cout << "\t Cell " << i << " Population: " << tmp << '\n';
				}
			}
			*/

			//reset collision flag variable for each particle
			for (std::vector<Particle>::iterator it = particles.begin(); it!=particles.end(); ++it) {
				it->collisionFlag = false;
			}
	
			grid.resetGrid();

			t+=p->dt; //iterate to next time step
			count++;
		}
		gettimeofday(&end, NULL);
		timeCpu=((end.tv_sec + end.tv_usec*0.000001) - (start.tv_sec + start.tv_usec*0.000001));
	}

	timeGpu = 0.0;
	/* PARALLEL IMPLEMENTATION */
	if (gpu) {
		double *positionXGpu = new double[p->numParticles];
		double *positionYGpu = new double[p->numParticles];
		timeGpu = picFloatGpu(positionX, positionY, velocityX, velocityY, positionXGpu, positionYGpu, p);	

		if (cpu) {
			//checkResults(particles, positionXGpu, positionYGpu);
			checkError(particles, positionXGpu, positionYGpu, p);
		}
	}

	printf("\nBlock size: %d\n", p->blockSize);
	printf("Problem size: %d x %d\n", p->numParticlesX, p->numParticlesY);
	//printf("Initialisation took %f seconds\n", initTime);
	printf("CPU took %f seconds\n", timeCpu);
	printf("GPU took %f seconds\n", timeGpu);
}	


/**
 * \brief Function to initialise the positions and velocities of a vector of Particle objects
 * They have uniform spatial density and maxwellian velocity with an added drift term
 */
void initialiseParticles(std::vector<Particle> &particles, std::shared_ptr<Params> p) {
	//creating rng to produce normally distributed RVs with mu=0, var=1
	//std::default_random_engine gen(time(NULL));
	std::default_random_engine gen(1234);
	std::normal_distribution<double> rand{0.0,1.0};

	double xCoord, yCoord;
	double xOffset = 0.5*(double)(p->xDim/p->nx);
	double yOffset = 0.5*(double)(p->yDim/p->ny);
	Point tmp;
	//fill particle positions and thermal velocities
	for (auto i=0; i<p->numParticlesY; i++) {
		//particles are uniformly distributed across the domain
		yCoord = ((double)p->yDim/p->numParticlesY)*(double)i + yOffset;
		for (auto j=0; j<p->numParticlesX; j++) {
			xCoord = ((double)p->xDim/p->numParticlesX)*(double)j + xOffset;
			particles.emplace_back(Point{xCoord, yCoord}, p);

			//thermal velocity assigned with Maxwellian distribution
			tmp.x = p->vtx * rand(gen);
			tmp.y = p->vty * rand(gen);
			particles[i*p->numParticlesX + j].setVelocity(tmp); 
		}
	}	

	//add drift velocity 
	double driftSumX = 0.0, driftSumY = 0.0;
	for (std::vector<Particle>::iterator it = particles.begin(); it!=particles.end(); ++it) {
		tmp = it->getVelocity();
		driftSumX += tmp.x;
		driftSumY += tmp.y;
	}

	double var = 1.0/(double)p->numParticles;
	driftSumX = var*driftSumX - p->vdx;
	driftSumY = var*driftSumY - p->vdy;

	for (std::vector<Particle>::iterator it = particles.begin(); it!=particles.end(); ++it) {
		tmp = it->getVelocity();	
		tmp.x -= driftSumX;
		tmp.y -= driftSumY;
		it->setVelocity(tmp);
	}
}

/**
 * \brief Function to store which particles are in each cell of the grid
 * 
 * \param particles	std::vector containing each particle from the CPU
 * \param grid		Grid object containing the cellTracker vector which we will write to
 * \param p		smart pointer to Params object containing simulation parameters
 */
void cellTracking(std::vector<Particle> particles, Grid& grid, std::shared_ptr<Params> p) {
	for (int i=0; i<p->numParticles; i++) {
		int j = particles[i].currentCell.x;
		int k = particles[i].currentCell.y;

		int index = j + k * p->ny;
		
		grid.cellTracker[index].push_back(i);
	}
}

/** 
 * \brief Function to simulate collisions with nearby particles
 *
 * \param gen		Mersenne twister RNG
 * \param particles	std::vector containing Particle objects
 * \param p		Smart pointer to Params struct which contains simulation variables
 * \param grid		Grid class instance to obtain cellTracker vector
 */
void collisions(std::mt19937& gen, std::vector<Particle> &particles, std::shared_ptr<Params> p, Grid& grid) {
	//loop over all particles
	for (int i=0; i<p->numParticles; i++) {
		double tmpX, tmpY;
		double nearest = 1e8;
		double distance;
		int target = -1;
	
		int j = particles[i].currentCell.x;
		int k = particles[i].currentCell.y;
		int index = j + k * p->ny;

		double posX = particles[i].position.x;
		double posY = particles[i].position.y;
	
		//loop over all particles in the same cell
		for (int l=0; l<grid.cellTracker[index].size(); l++) {
			int id = grid.cellTracker[index][l];	
			//a particle cannot collide with itself
			if (i != id) {
				tmpX = particles[id].getPosition().x;	
				tmpY = particles[id].getPosition().y;	
				distance = sqrt((posX-tmpX)*(posX-tmpX) + (posY-tmpY)*(posY-tmpY));
				if (distance < nearest) {
					nearest = distance;
					target = id;
				}
			}
		}

		//we only wish to calculate each collision once so we choose to use the lower index
		if (target < i) {
			target = -1;
		}

		if (target != -1) {
			//generate collision probability, q
			double q = 0.2;

			//if probability > random number, collision occurs
			std::uniform_real_distribution<double> dist{0.0, 1.0};
			double r = dist(gen);	
			
			if (q > r) {
				//generate two maxwellian random variables, using Bird's method
				double max1 = 2*(dist(gen) + dist(gen) + dist(gen) - 1.5);
				double max2 = 2*(dist(gen) + dist(gen) + dist(gen) - 1.5);
		
				Point tmp;
				
				//set colliding particle velocity
				tmp.x = max1;
				tmp.y = max2;
				particles[i].setVelocity(tmp);
			
				//set target particle velocity
				tmp.x = -max1;
				tmp.y = -max2;
				particles[target].setVelocity(tmp);
			}
		}	
	}	
}

/**
 * \brief Function to check if the final positions of the CPU and GPU output agree
 * 
 * \param particles	std::vector containing each particle from the CPU
 * \param xGpu		double array containing x-coordinates from the GPU
 * \param yGpu		double array containing y-coordinates from the GPU
 */
void checkResults(std::vector<Particle> &particles, double* xGpu, double *yGpu) {
	int i = 0;
	std::cout << "Checking results.\n";
	for (std::vector<Particle>::iterator it=particles.begin(); it!=particles.end(); ++it) {
		Point tmp = it->position;
		if (fabs(tmp.x - xGpu[i]) > 1e-4 || fabs(tmp.y - yGpu[i]) > 1e-4) {
			std::cout << "Particle " << i+1 << ": CPU and GPU results don't agree\n";
			std::cout << "Serial: (" << tmp.x << ',' << tmp.y << ")\t (" << xGpu[i] << ',' << yGpu[i] << ")\n";
		}
		i++;
	}
}


/**
 * \brief Function to check if the final positions of the CPU and GPU output agree
 * 
 * \param particles	std::vector containing each particle from the CPU
 * \param xGpu		double array containing x-coordinates from the GPU
 * \param yGpu		double array containing y-coordinates from the GPU
 */
void checkError(std::vector<Particle> &particles, double* xGpu, double *yGpu, std::shared_ptr<Params> p) {
	int i = 0;
	double errX = 0, errY= 0;
	double localError = 0;
	double error = 0;
	std::cout << "\nCalculating average error.\n";
	for (std::vector<Particle>::iterator it=particles.begin(); it!=particles.end(); ++it) {
		Point tmp = it->position;

		//x coordinate error
		errX = tmp.x-xGpu[i];
		if (fabs(errX) > 0.1*p->nx) {
			(tmp.x > xGpu[i]) ? (errX=xGpu[i]) : (errX=tmp.x);
		}

		//y coordinate error
		errY = tmp.y-yGpu[i];
		if (fabs(errY) > 0.1*p->ny) {
			(tmp.y > yGpu[i]) ? (errY=yGpu[i]) : (errY=tmp.y);	
		}

		//absolute error
		localError = sqrt(errX*errX + errY*errY);
		error += localError;
		i++;
	}
	error = error/(p->numParticles);
	std::cout << "Error = " << error << '\n';
}
