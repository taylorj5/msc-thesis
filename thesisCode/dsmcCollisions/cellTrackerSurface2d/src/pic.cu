#include <memory>
#include <stdio.h>	
#include <iostream>

#include "params.h"
#include "particle.h"
#include "pic.h"
#include <cufft.h>
#include <cufftXt.h>
#include <unistd.h>
#include <curand_kernel.h>	

//declaring variables in constant memory
__device__ __constant__ int xDimC;		//size of grid in x dimension
__device__ __constant__ int yDimC;		//size of grid in y dimension
__device__ __constant__ int nxC;		//num grid points in x direction
__device__ __constant__ int nyC;		//num grid points in y direction
__device__ __constant__ int npxC;		//num particles in x direction
__device__ __constant__ int npyC;		//num particles in y direction
__device__ __constant__ int npC;		//num particles
__device__ __constant__ int cellMaxC;		//time increment between iterations
__device__ __constant__ double chargeC;		//charge of superparticle
__device__ __constant__ double massC;		//mass of superparticle
__device__ __constant__ double dxC;		//size of cell in x direction
__device__ __constant__ double dyC;		//size of cell in y direction
__device__ __constant__ double cellAreaC;	//area of one cell
__device__ __constant__ double qmC;		//ration of charge to mass of particle
__device__ __constant__ double dtC;		//time increment between iterations


#define cudaErrorCheck(err)\
	if (err != cudaSuccess){ \
		std::cerr << "ERROR:" << cudaGetErrorString(err) << '\n'; \
		exit (-1); \
	}

/**
 * \brief  Function to perform double-precision atomic addition in CUDA.
 * Necessary to use atomic add for doubles with compute capability  < 6.0
 *
 * \param	address of value to add to
 * \param	value to add to the variable
 */
__device__ double atomicAddDouble(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*) address;
	unsigned long long int old = *address_as_ull;
	unsigned long long int assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}


/**
 * \brief Cuda kernel to calculate current cell of each particle on the gpu
 *
 * \param positionX	Array containing x-coordinates of all particles
 * \param positionY	Array containing y-coordinates of all particles
 * \param cellX 	Array to store current cell in x direction of each particle
 * \param cellY 	Array to store current cell in y direction of each particle
 */
__global__ void currentCell(double* positionX, double* positionY, int* cellX, int* cellY) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;

	//loading size of particles vector from grid
	int n = npxC*npyC;

	//initialising shared memory and loading array into it
	extern __shared__ double sdata[];
	sdata[2*tid] 	= (idx < n) ? positionX[idx] : 0;
	sdata[2*tid+1]	= (idx < n) ? positionY[idx] : 0;
	__syncthreads();

	if (idx < n) {
		cellX[idx] = (int) (positionX[idx]/dxC); //integer arithmetic
		cellY[idx] = (int) (positionY[idx]/dyC);
	}
}


/*
 * \brief Cuda kernel to assign particle charges to the nearest grid points
 *
 * \param positionX	Array containing x-coordinates of all particles
 * \param positionY	Array containing y-coordinates of all particles
 * \param cellX 	Array containing current cell in x direction of each particle
 * \param cellY 	Array containing current cell in y direction of each particle
 * \param density	Array to store the charge density of each gridpoint 
 */
__global__ void chargeAssignment(double* positionX, double* positionY, int* cellX, int* cellY, double* density) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	double invArea = 1.0/cellAreaC;
	int n = npxC*npyC;
	int y = nyC;

	extern __shared__ double sdata[];	
	sdata[4*tid] 	= (idx < n) ? positionX[idx] : 0;
	sdata[4*tid+1]	= (idx < n) ? positionY[idx] : 0;
	sdata[4*tid+2] 	= (idx < n) ? cellX[idx] : 0;
	sdata[4*tid+3] 	= (idx < n) ? cellY[idx] : 0;
	__syncthreads();

	//adding to charge density of nearby points
	if (idx < n) {
		int xCell = cellX[idx];
		int yCell = cellY[idx];

		double area1 = ((positionX[idx] - cellX[idx]*dxC) * (positionY[idx] - cellY[idx]*dyC))*invArea;
		double area2 = (((cellX[idx]*dxC + dxC) - positionX[idx]) * (positionY[idx] - cellY[idx]*dyC))*invArea;
		double area3 = ((positionX[idx] - cellX[idx]*dxC) * ((cellY[idx]*dyC + dyC) - positionY[idx]))*invArea;
		double area4 = (((cellX[idx]*dxC + dxC) - positionX[idx]) * ((cellY[idx]*dyC + dyC) - positionY[idx]))*invArea;

		//way with integer logic
		density[xCell*y + yCell] 				+= (area1 * chargeC);
		density[((xCell+1)%nxC)*y + yCell] 			+= (area2 * chargeC);
		density[(xCell)*y + ((yCell+1)%nyC)] 			+= (area3 * chargeC);
		density[((xCell+1)%nxC)*y + ((yCell+1)%nyC)]		+= (area4 * chargeC);

		//using atomic add
		/*
		atomicAddDouble(&density[xCell*y + yCell], area1*chargeC);
		atomicAddDouble(&density[((xCell+1)%nxC)*y + yCell], area2*chargeC);
		atomicAddDouble(&density[xCell*y + ((yCell+1)%nyC)], area3*chargeC);
		atomicAddDouble(&density[((xCell+1)%nxC)*y + ((yCell+1)%nyC)], area4*chargeC);
		__syncthreads();
		*/
	}
}


/**
 * \brief Kernel to solve the Poisson equation in Fourier space
 *
 * \param arr	Array containing output of forward R2C Fourier transform
 * \param nyh	Physical y-dimension of Fourier transform output (not logical size due to Hermitian symmetry)
 */
__global__ void fftPoissonSolver(cufftDoubleComplex* arr, const int nyh) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int n = nxC*nyh;

	//initialising shared memory and loading array into it
	extern __shared__ double sdata[];
	sdata[2*tid] 	= (idx < n) ? arr[idx].x : 0;
	sdata[2*tid+1]	= (idx < n) ? arr[idx].y : 0;
	__syncthreads();


	double pi = 3.141592654f;

	int i, j;
	int II, JJ;
	double k1, k2;

	if (idx < n) {
		i = idx/nyh;
		j = idx%nyh;

		//setting II and JJ
		if (2*i < nxC){
			II = i;
		} else {
			II = i - nxC;
		}
		if (2*j < nyh) {
			JJ = j;
		} else {
			JJ = j - nyh;
		}

		//setting wavevector
		k1 = 2*pi*II;
		k2 = 2*pi*JJ;

		double fact = k1*k1 + k2*k2;
		double invFact = __fdividef(-1.0, fact);
		if (fabsf(fact) < 1e-14) {
			arr[idx].x = 0.0;
			arr[idx].y = 0.0;
		} else {
			arr[idx].x *= invFact;	
			arr[idx].y *= invFact;
		}
	}
}


/**
 * \brief Simple kernel to copy memory from device array to device array
 * Avoids latency of devicetodevice memcpy API calls
 *
 * \param arr	array containing inverse FFT output
 */
__global__ void copyD2D(double *dest, double *src) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int n = nxC*nyC;
	
	if (idx < n) {
		dest[idx] = src[idx];
	}
}


/**
 * \brief Kernel to normalise the output of the Fourier transform
 *
 * \param arr	array containing inverse FFT output
 */
__global__ void normaliseTransform(double *arr) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int n = nxC*nyC;

	//initialising shared memory and loading array into it
	extern __shared__ double sdata[];
	sdata[tid] = (idx < n) ? arr[idx] : 0;
	__syncthreads();

	if (idx < n) {
		double norm = __fdividef(-1.0, n);
		arr[idx] *= norm;
	}
}


/**
 * \brief Kernel to compute the electric field given the electric potential
 *
 * \param fieldX	double array to store the values of the electric field in x-direction
 * \param fieldY	double array to store the values of the electric field in y-direction
 * \param potential	double array in which the electric potential is stored
 */
__global__ void computeElectricField(double* fieldX, double* fieldY, double* potential) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int n = nxC*nyC;

	//initialising shared memory and loading array into it
	extern __shared__ double sdata[];
	sdata[tid] = (idx < n) ? potential[idx] : 0;
	__syncthreads();

	double divisorX = 2*dxC; 
	double divisorY = 2*dyC;

	if (idx < n) {
		double i = idx%nyC;
		double j = idx/nyC;
	
		//setting x component of electric field
		if (i==0){
			fieldX[idx] = (potential[idx+(nxC-1)]-potential[idx+1])*divisorX;
		}
		else if (i==(nxC-1)) {
			fieldX[idx] = (potential[idx-1]-potential[idx-(nxC-1)])*divisorX;
		}
		else {
			fieldX[idx] = (potential[idx-1]-potential[idx+1])*divisorX;
		}

		//setting y component of electric field
		if (j==0) {
			fieldY[idx] = (potential[idx+nyC*(nxC-1)]-potential[idx+nyC])*divisorY;
		}
		else if (j==(nyC-1)) {
			fieldY[idx] = (potential[idx-nyC]-potential[idx-nyC*(nxC-1)])*divisorY;
		}
		else {
			fieldY[idx] = (potential[idx-nyC]-potential[idx+nyC])*divisorY;
		}
	}
}


/**
 * \brief Kernel to compute the force acting on each particle
 *
 * \param positionX 	double array containing x-coordinate of each particle
 * \param positionY	double array containing y-coordinate of each particle
 * \param cellX		int array containing current cell in x-direction of each particle
 * \param cellY		int array containing current cell in y-direction of each particle
 * \param fieldX	double array containing x-component of electric field at each grid point
 * \param fieldY	double array containing y-component of electric field at each grid point
 * \param forceX	double array to store x-component of force on each particle
 * \param forceY	double array to store y-component of force on each particle
 */
__global__ void forceCalculation(double* positionX, double* positionY, int* cellX, int* cellY, double* fieldX, double* fieldY, double* forceX, double* forceY) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	double invArea = 1.0/cellAreaC;
	int n = npxC*npyC;
	int y = nyC;

	//initialising shared memory and loading arrays into it
	extern __shared__ double sdata[];	
	sdata[4*tid] 	= (idx < n) ? positionX[idx] : 0;
	sdata[4*tid+1]	= (idx < n) ? positionY[idx] : 0;
	sdata[4*tid+2] 	= (idx < n) ? cellX[idx] : 0;
	sdata[4*tid+3] 	= (idx < n) ? cellY[idx] : 0;
	__syncthreads();

	//computing force acting on each particle
	if (idx < n) {
		double tmp = 0.0;
		int xCell = cellX[idx];
		int yCell = cellY[idx];

		double area1 = ((positionX[idx] - xCell*dxC) * (positionY[idx] - yCell*dyC))*invArea;
		double area2 = (((xCell*dxC + dxC) - positionX[idx]) * (positionY[idx] - yCell*dyC))*invArea;
		double area3 = ((positionX[idx] - xCell*dxC) * ((yCell*dyC + dyC) - positionY[idx]))*invArea;
		double area4 = (((xCell*dxC + dxC) - positionX[idx]) * ((yCell*dyC + dyC) - positionY[idx]))*invArea;

		//computing X component of the force
		//with integer arithmetic
		forceX[idx] = 0.0;

		//integer arithmetic
		tmp += area1 * fieldX[xCell*y + yCell];
		tmp += area2 * fieldX[((xCell+1)%nxC)*y + yCell];
		tmp += area3 * fieldX[xCell*y + ((yCell+1)%y)];
		tmp += area4 * fieldX[((xCell+1)%nxC)*y + (yCell+1)%y];
		forceX[idx] = tmp*qmC;

		//computing Y component of the force
		//using integer arithmetic
		forceY[idx] = 0.0;
		tmp = 0.0;

		tmp += area1 * fieldY[xCell*y + yCell];
		tmp += area2 * fieldY[((xCell+1)%nxC)*y + yCell];
		tmp += area3 * fieldY[xCell*y + (yCell+1)%y];
		tmp += area4 * fieldY[((xCell+1)%nxC)*y + (yCell+1)%y];
		forceY[idx] = tmp*qmC;
	}
}


/**
 * \brief Kernel to update the particle positions given the force acting on them
 *
 * \param positionX	double array containing x-component of position
 * \param positionY	double array containing y-component of position
 * \param velocityX	double array containing x-component of velocity
 * \param velocityY	double array containing y-component of velocity
 * \param forceX	double array containing x-component of force
 * \param forceY	double array containing y-component of force
 * \param forceOldX	double array containing x-component of force at previous time-step
 * \param forceOldY	double array containing y-component of force at previous time-step
 * \param cellX		int array containing current cell in x-direction of particle
 * \param cellY		int array containing current cell in y-dircetion of particle
 */
__global__ void positionUpdate(double* positionX, double* positionY, double* velocityX, double* velocityY, double* forceX, double* forceY, double* forceOldX, double* forceOldY, int* cellX, int* cellY) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int n = npC;	

	//initialising shared memory and loading arrays into it
	extern __shared__ double sdata[];	
	sdata[4*tid] 	= (idx < n) ? positionX[idx] : 0;
	sdata[4*tid+1]	= (idx < n) ? positionY[idx] : 0;
	sdata[4*tid+2] 	= (idx < n) ? velocityX[idx] : 0;
	sdata[4*tid+3] 	= (idx < n) ? velocityY[idx] : 0;
	__syncthreads();
	
	if (idx < n) {
		//compute velocity at half time step
		double vxh = velocityX[idx] + 0.5*dtC*forceOldX[idx];
		double vyh = velocityY[idx] + 0.5*dtC*forceOldY[idx];
			
		//update position
		positionX[idx] += vxh*dtC;
		positionY[idx] += vyh*dtC;
		
		//correct position to ensure 0 < posX/posY < xDim/yDim
		if (positionX[idx] > xDimC) {positionX[idx] -= xDimC;}
		else if (positionX[idx] < 0) {positionX[idx] += xDimC;}
		
		if (positionY[idx] > yDimC) {positionY[idx] -= yDimC;}
		else if (positionY[idx] < 0) {positionY[idx] += yDimC;}

		//update velocity
		velocityX[idx] = vxh + 0.5*dtC*forceX[idx];
		velocityY[idx] = vyh + 0.5*dtC*forceY[idx];
		
		//update current cell
		//cellX[idx] = (int)(positionX[idx]/dxC);
		//cellY[idx] = (int)(positionY[idx]/dyC);
	}
}


/**
 * \brief Kernel to initialise cuRAND RNG
 *
 * \param state		random number generator state
 */
__global__ void initialiseGenerator(curandState *state) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int n = npC;	

	if (idx < n) {
		//setting up each thread with same seed but different sequence and no offset
		curand_init(1234, idx, 0, &state[idx]);
	}
}


/**
 * \brief Kernel for grid cells to determine which particles they store
 *
 * \param cellArray	array to store particles residing in each cell
 * \param cellX		array containing current cell in x-direction of particle
 * \param cellY		array containing current cell in y-direction of particle
 */
__global__ void cellTracking(int* cellX, int* cellY, cudaSurfaceObject_t surf) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int n = nxC*nyC;

	//initialising shared memory and loading arrays into it
	extern __shared__ double sdata[];
	sdata[2*tid] 	= (idx < n) ? cellX[idx] : 0;
	sdata[2*tid+1]	= (idx < n) ? cellY[idx] : 0;
	__syncthreads();

	if (idx < n) {
		int i = idx/nyC; 
		int j = idx%nyC; 
		int row = (i + j*nyC);
		int col = 0;
		
		//loop over all particles and sort into corresponding cells
		for (int k=0; k<npC; k++) {
			if (cellX[k] == i && cellY[k] == j) {
				surf2Dwrite(k, surf, row*sizeof(int), col, cudaBoundaryModeTrap);
				col++;
			}
		}
		__syncthreads();
	}
}


/**
 * \brief Kernel to compute collisions between particle pairs in the same cell
 *
 * \param state		cuRAND state from which RNs are generated
 * \param cellX		array containing current cell in x-direction of particle
 * \param cellY		array containing current cell in y-direction of particle
 * \param positionX	double array containing x-component of position
 * \param positionY	double array containing y-component of position
 * \param velocityX	double array containing x-component of velocity
 * \param velocityY	double array containing y-component of velocity
 * \param surf		surface memory array containing particles residing in each cell
 */
__global__ void collisions(curandState *state, int* cellX, int* cellY, double* positionX, double* positionY, double* velocityX, double* velocityY, cudaSurfaceObject_t surf) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int n = npC;
	double p, r;
	double tmpX, tmpY;
	double distance, nearest=1e8;
	int target = -1;

	//initialising shared memory and loading arrays into it
	extern __shared__ double sdata[];	
	sdata[5*tid] 	= (idx < n) ? cellX[idx] : 0;
	sdata[5*tid+1]	= (idx < n) ? cellY[idx] : 0;
	sdata[5*tid+2] 	= (idx < n) ? velocityX[idx] : 0;
	sdata[5*tid+3]	= (idx < n) ? velocityY[idx] : 0;
	sdata[5*tid+4]	= (idx < n) ? positionX[idx] : 0;
	__syncthreads();

	if (idx < n) {
		int i = cellX[idx];
		int j = cellY[idx];

		double posX = positionX[idx];
		double posY = positionY[idx];

		//determine storage point of particles in the given cell
		int pos = i*nyC + j;

		//we loop through all particles in the same cell to create collision pairs
		for (int k=0; k<cellMaxC; k++) {
			int index;
			surf2Dread(&index, surf, pos*sizeof(int), k);
			if (index == -1) {
				break;
			}
			//cannot collide with self
			if (index != idx) {
				tmpX = positionX[index];
				tmpY = positionY[index];
				distance = sqrt((posX-tmpX)*(posX-tmpX) + (posY-tmpY)*(posY-tmpY));
				if (distance < nearest) {
					target = index;
					nearest = distance;
				}	
			}
		}

		//we only wish to calculate each collision once so we choose to use the lower index
		if (target < idx) {
			//collFlag[target] = -1;
			target = -1;
		}

		if (target != -1) {
			//generate collision probability, p
			p = 0.2;
			
			//generate random number, r
			curandState localState = state[idx];
			r = curand_uniform_double(&localState);
		
			//if probability > random number, collision occurs
			if (p > r) {
				//we replace the velocities from the maxwellian distribution (sampling method taken from Bird)
				double max1 = 2*(curand_uniform_double(&localState) + curand_uniform_double(&localState) + curand_uniform_double(&localState) - 1.5);
				double max2 = 2*(curand_uniform_double(&localState) + curand_uniform_double(&localState) + curand_uniform_double(&localState) - 1.5);
				velocityX[idx] = max1;
				velocityY[idx] = max2;
				velocityX[target] = -max1;
				velocityY[target] = -max2;
			}
		}
	}
}


/**
 * \brief Kernel to reset the surface array to -1
 *
 * \param cellArray	array to store particles residing in each cell
 */
__global__ void surfaceMemoryWrite(const int N, cudaSurfaceObject_t surf) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int n = nxC*nyC;

	if (idx < n) {
		for (int i=0; i<cellMaxC; i++) {
			surf2Dwrite(N, surf, idx*sizeof(int), i);
		}
	}
}


/**
 * \brief Function to declare device variables and call relevant kernels on the gpu
 *
 * \param positionX 
 * \param positionY
 * \param velocityX
 * \param velocityY
 * \param xResult
 * \param yResult
 * \param p		
 */
double picFloatGpu(double* positionX, double *positionY, double* velocityX, double* velocityY, double *xResult, double *yResult, std::shared_ptr<Params> p) {
	cudaSetDevice(0);
	//cudaSetDevice(1);

	//creating cuda streams
	cudaStream_t stream[2];
	for (int i=0; i<2; i++) {
		cudaStreamCreate(&stream[i]);
	}

	//printf("starting on gpu\n");

	//initialise required variables for time measurement
	cudaEvent_t start, finish;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);
	cudaEventRecord(start, 0); 

	//declaring variables to be stored in constant memory
	const int xDim = p->xDim;
	const int yDim = p->yDim;
	const int nx = p->nx;
	const int ny = p->ny;
	const int npx = p->numParticlesX;
	const int npy = p->numParticlesY;
	const int np = npx*npy;
	const int cellMax = (np > nx*ny) ? (np/(nx*ny))*4 : 6;
	const double charge = p->electronCharge;
	const double mass = p->mass;
	const double dx = p->dx;
	const double dy = p->dy;
	const double area = p->area;
	const double qm = p->electronCharge/p->mass;
	const double dt = p->dt;

	//grid configuration for particle kernels
	dim3 dimBlock(p->blockSize);
	dim3 dimBlockCells(32);
	dim3 dimGridParticles((np/dimBlock.x)+(!(np%dimBlock.x)?0:1));

	//grid configuration for grid kernels
	int ng = nx*ny;
	dim3 dimGridCells((ng/dimBlockCells.x)+(!(ng%dimBlockCells.x)?0:1));
	dim3 dimGridAll((cellMax*ng/dimBlock.x)+(!((cellMax*ng)%dimBlock.x)?0:1));

	//shared memory configuration
	int smem1 = dimBlock.x*sizeof(double);		//size of shared memory in each block (1 double per thread)
	int smem2 = dimBlock.x*2*sizeof(double); 	//size of shared memory in each block (2 variables per thread)
	int smem4 = dimBlock.x*4*sizeof(double); 	//size of shared memory in each block (4 variables per thread)
	int smem5 = dimBlock.x*5*sizeof(double); 	//size of shared memory in each block (5 variables per thread)

	//copying data to constant memory
	cudaMemcpyToSymbolAsync(xDimC, 		&xDim, 		sizeof(int), 		0,	cudaMemcpyHostToDevice, 	stream[0]);
	cudaMemcpyToSymbolAsync(yDimC, 		&yDim, 		sizeof(int),		0,	cudaMemcpyHostToDevice, 	stream[1]);
	cudaMemcpyToSymbolAsync(nxC, 		&nx, 		sizeof(int),		0,	cudaMemcpyHostToDevice, 	stream[0]);
	cudaMemcpyToSymbolAsync(nyC, 		&ny, 		sizeof(int),		0,	cudaMemcpyHostToDevice, 	stream[1]);
	cudaMemcpyToSymbolAsync(npxC, 		&npx,		sizeof(int),		0, 	cudaMemcpyHostToDevice,		stream[0]);
	cudaMemcpyToSymbolAsync(npyC, 		&npy,	 	sizeof(int),		0,	cudaMemcpyHostToDevice, 	stream[1]);
	cudaMemcpyToSymbolAsync(npC, 		&np,	 	sizeof(int),		0,	cudaMemcpyHostToDevice, 	stream[0]);
	cudaMemcpyToSymbolAsync(cellMaxC, 	&cellMax,	sizeof(int),		0,	cudaMemcpyHostToDevice,		stream[1]);
	cudaMemcpyToSymbolAsync(chargeC, 	&charge,	sizeof(double),		0,	cudaMemcpyHostToDevice, 	stream[0]);
	cudaMemcpyToSymbolAsync(massC, 		&mass,		sizeof(double),		0,	cudaMemcpyHostToDevice,		stream[1]);
	cudaMemcpyToSymbolAsync(dxC, 		&dx, 		sizeof(double), 	0, 	cudaMemcpyHostToDevice, 	stream[0]);
	cudaMemcpyToSymbolAsync(dyC, 		&dy, 		sizeof(double),		0, 	cudaMemcpyHostToDevice, 	stream[1]); 
	cudaMemcpyToSymbolAsync(cellAreaC, 	&area, 		sizeof(double),		0, 	cudaMemcpyHostToDevice, 	stream[0]); 
	cudaMemcpyToSymbolAsync(qmC,	 	&qm, 		sizeof(double),		0, 	cudaMemcpyHostToDevice, 	stream[1]); 
	cudaMemcpyToSymbolAsync(dtC,	 	&dt, 		sizeof(double),		0, 	cudaMemcpyHostToDevice, 	stream[0]); 

	//pointers to device memory
	double *positionXGpu, *positionYGpu;
	int *currentCellXGpu, *currentCellYGpu;
	double *velocityXGpu, *velocityYGpu;
	double *forceXGpu, *forceYGpu;
	double *forceXOldGpu, *forceYOldGpu;
	double *chargeDensityGpu;
	double *electricPotentialGpu;
	double *electricFieldXGpu, *electricFieldYGpu;

	//allocating device memory for data arrays
	cudaMalloc((void**) &positionXGpu,		np*sizeof(double)); 	//memory for particle positions
	cudaMalloc((void**) &positionYGpu, 		np*sizeof(double)); 	//memory for particle positions
	cudaMalloc((void**) &velocityXGpu, 		np*sizeof(double)); 	//memory for velocity of particles
	cudaMalloc((void**) &velocityYGpu,		np*sizeof(double)); 	//memory for velocity of particles
	cudaMalloc((void**) &currentCellXGpu, 		np*sizeof(int)); 	//memory for current cell of each particle
	cudaMalloc((void**) &currentCellYGpu, 		np*sizeof(int)); 	//memory for current cell of each particle
	cudaMalloc((void**) &forceXGpu, 		np*sizeof(double)); 	//memory for force acting on particles
	cudaMalloc((void**) &forceYGpu, 		np*sizeof(double)); 	//memory for force acting on particles
	cudaMalloc((void**) &forceXOldGpu, 		np*sizeof(double)); 	//memory for force acting on particles at previous timestep
	cudaMalloc((void**) &forceYOldGpu, 		np*sizeof(double)); 	//memory for force acting on particles at previous timestep
	cudaMalloc((void**) &chargeDensityGpu, 		nx*ny*sizeof(double)); 	//memory for charge density at all grid points
	cudaMalloc((void**) &electricPotentialGpu, 	nx*ny*sizeof(double)); 	//memory for electric potential at all grid points
	cudaMalloc((void**) &electricFieldXGpu, 	nx*ny*sizeof(double)); 	//memory for electric field at all grid points
	cudaMalloc((void**) &electricFieldYGpu, 	nx*ny*sizeof(double)); 	//memory for electric field at all grid points


	//creating cuda array to bind to surface
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaArray *cellTracker;
	cudaMallocArray(&cellTracker, &channelDesc, ng, cellMax, cudaArraySurfaceLoadStore);	//memory for keeping track of which cell particles are in

	//binding cellTracker to surface using surface object
	cudaSurfaceObject_t surface;
	cudaResourceDesc surfRes;
	memset(&surfRes, 0, sizeof(cudaResourceDesc));
	surfRes.resType = cudaResourceTypeArray;
	surfRes.res.array.array = cellTracker;

	//creating surface object and writing to it
	cudaCreateSurfaceObject(&surface, &surfRes);
	surfaceMemoryWrite<<<dimGridAll,dimBlock,0,stream[0]>>>(-1, surface);	
	cudaDeviceSynchronize();

	//copying particle data from host	
	cudaMemcpyAsync(positionXGpu,	positionX, 	np*sizeof(double), 	cudaMemcpyHostToDevice, 	stream[0]);
	cudaMemcpyAsync(positionYGpu,	positionY, 	np*sizeof(double), 	cudaMemcpyHostToDevice, 	stream[1]);
	cudaMemcpyAsync(velocityXGpu,	velocityX, 	np*sizeof(double), 	cudaMemcpyHostToDevice, 	stream[0]);
	cudaMemcpyAsync(velocityYGpu, 	velocityY,	np*sizeof(double), 	cudaMemcpyHostToDevice, 	stream[1]);

	//creating cuFFT plans
	cufftHandle planForward, planInverse;
	cufftCreate(&planForward);
	cufftCreate(&planInverse);
  	cufftPlan2d(&planForward, nx, ny, CUFFT_D2Z);
	cufftPlan2d(&planInverse, nx, ny, CUFFT_Z2D);

	//allocating memory for fft
	const int nyh = ny/2 + 1;
	double *in;
	cufftDoubleComplex *out;
	cudaMalloc((void**) &out, nx*nyh*sizeof(cufftDoubleComplex));

	//declaring variables for curand RNG
	curandState *states;
	cudaMalloc((void **)&states, np*sizeof(curandState));

	//initialise curand generator
	initialiseGenerator<<<dimGridParticles,dimBlock,smem2,stream[0]>>>(states);
		//setting current cells of each particle
		currentCell<<<dimGridParticles,dimBlock,smem2,stream[0]>>>(positionXGpu, positionYGpu, currentCellXGpu, currentCellYGpu);	
	
	//starting iteration
	double t = 0;
	//printf("Starting iteration on GPU.\n");
	while (t < p->tmax) {

		//interpolating charge to gridpoints
		chargeAssignment<<<dimGridParticles,dimBlock,smem4,stream[0]>>>(positionXGpu, positionYGpu, currentCellXGpu, currentCellYGpu, chargeDensityGpu);

		//forward R2C transform
		in = chargeDensityGpu;
		cufftExecD2Z(planForward, in, out);

		//computing poisson equation in Fourier space
		fftPoissonSolver<<<dimGridCells,dimBlockCells,smem2,stream[0]>>>(out, nyh);
		
		//inverse C2R transform and normalisation
		cufftExecZ2D(planInverse, out, in);
		normaliseTransform<<<dimGridCells, dimBlockCells, smem2, stream[0]>>>(in);

		//copy FFT Poisson output to electric potential array
		copyD2D<<<dimGridCells,dimBlockCells,0,stream[0]>>>(electricPotentialGpu, in);

		//computing electric field from electric potential
		computeElectricField<<<dimGridCells,dimBlockCells,smem1,stream[0]>>>(electricFieldXGpu, electricFieldYGpu, electricPotentialGpu);

		//computing force on each particle
		forceCalculation<<<dimGridParticles,dimBlock,smem4,stream[0]>>>(positionXGpu, positionYGpu, currentCellXGpu, currentCellYGpu, electricFieldXGpu, electricFieldYGpu, forceXGpu, forceYGpu);

		//updating particle positions
		positionUpdate<<<dimGridParticles,dimBlock,smem4,stream[0]>>>
			(positionXGpu, positionYGpu, velocityXGpu, velocityYGpu, forceXGpu, forceYGpu, forceXOldGpu, forceYOldGpu, currentCellXGpu, currentCellYGpu);

		//copy new force to forceOld arrays
		copyD2D<<<dimGridCells,dimBlock,0,stream[0]>>>(forceXOldGpu, forceXGpu);
		copyD2D<<<dimGridCells,dimBlock,0,stream[1]>>>(forceYOldGpu, forceYGpu);

		//store which particles are stored in each cell
		cellTracking<<<dimGridCells,dimBlockCells,smem2,stream[0]>>>(currentCellXGpu, currentCellYGpu, surface);
		cudaDeviceSynchronize();

		//simulate collisions
		collisions<<<dimGridParticles,dimBlock,smem5,stream[0]>>>(states,currentCellXGpu, currentCellYGpu, positionXGpu, positionYGpu, velocityXGpu, velocityYGpu, surface);

		//reset cellTracker to -1
		surfaceMemoryWrite<<<dimGridAll,dimBlock,0,stream[0]>>>(-1, surface);

		t += dt;
	}

	
	//copying results back to the host
	cudaMemcpyAsync(xResult, positionXGpu, np*sizeof(double), cudaMemcpyDeviceToHost, stream[0]);
	cudaMemcpyAsync(yResult, positionYGpu, np*sizeof(double), cudaMemcpyDeviceToHost, stream[1]);

	//stop timer
	cudaEventRecord(finish, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(finish);
	cudaEventElapsedTime(&elapsedTime, start, finish);

	//destroying cuda streams
	for (int i=0; i<2; i++) {
		cudaStreamDestroy(stream[i]);
	}

	//freeing dynamically allocated memory
	cudaFree(positionXGpu), cudaFree(positionYGpu);
	cudaFree(velocityXGpu), cudaFree(velocityYGpu);
	cudaFree(currentCellXGpu), cudaFree(currentCellYGpu);
	cudaFree(forceXGpu), cudaFree(forceYGpu);
	cudaFree(forceXOldGpu), cudaFree(forceYOldGpu);
	cudaFree(chargeDensityGpu);
	cudaFree(electricPotentialGpu);
	cudaFree(electricFieldXGpu), cudaFree(electricFieldYGpu);

	//return elapsed time in seconds
	return (elapsedTime/1000);
}
