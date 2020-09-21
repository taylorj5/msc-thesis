#include <memory>
#include <stdio.h>	
#include <iostream>

#include "params.h"
#include "particle.h"
#include "pic.h"
#include <cufft.h>
#include <cufftXt.h>
#include <unistd.h>

//declaring variables in constant memory
__device__ __constant__ int xDimC;		//size of grid in x dimension
__device__ __constant__ int yDimC;		//size of grid in y dimension
__device__ __constant__ int nxC;		//num grid points in x direction
__device__ __constant__ int nyC;		//num grid points in y direction
__device__ __constant__ int npxC;		//num particles in x direction
__device__ __constant__ int npyC;		//num particles in y direction
__device__ __constant__ int npC;		//num particles in y direction
__device__ __constant__ float chargeC;		//charge of superparticle
__device__ __constant__ float massC;		//mass of superparticle
__device__ __constant__ float dxC;		//size of cell in x direction
__device__ __constant__ float dyC;		//size of cell in y direction
__device__ __constant__ float cellAreaC;	//area of one cell
__device__ __constant__ float qmC;		//ration of charge to mass of particle
__device__ __constant__ float dtC;		//time increment between iterations

#define CHECK_ERROR(err)\
	if (err != cudaSuccess){ \
		std::cerr << "ERROR:" << cudaGetErrorString(err) << '\n'; \
		exit (-1); \
	}

#define CHECK_LAST_ERROR(err)\
	{cudaError_t = cudaGetLastError(); \
		if (err != cudaSuccess) {\
			std::cerr << cudaGetErrorString(err) << '\n'; \
			exit(-1); \
		}\
	}

/**
 * \brief Cuda kernel to calculate current cell of each particle on the gpu
 *
 * \param positionX	Array containing x-coordinates of all particles
 * \param positionY	Array containing y-coordinates of all particles
 * \param cellX 	Array to store current cell in x direction of each particle
 * \param cellY 	Array to store current cell in y direction of each particle
 */
__global__ void currentCell(float* positionX, float* positionY, int* cellX, int* cellY) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	//int idy = blockDim.y * blockIdx.y + threadIdx.y;
	int tid = threadIdx.x;

	//loading size of particles vector from grid
	int n = npxC*npyC;

	//initialising shared memory and loading array into it
	extern __shared__ float sdata[];
	sdata[2*tid] 	= (idx < n) ? positionX[idx] : 0;
	sdata[2*tid+1]	= (idx < n) ? positionY[idx] : 0;
	__syncthreads();

	if (idx < n) {
		//cellX[idx] = __fdiv_rd(positionX[idx], dxC); //faster using float operations?
		//cellY[idx] = __fdiv_rd(positionY[idx], dyC);
		cellX[idx] = (int) (positionX[idx]/dxC); //integer arithmetic
		cellY[idx] = (int) (positionY[idx]/dyC);
		//printf("(%f, %f) -> (%d, %d)\n", positionX[idx], positionY[idx], cellX[idx], cellY[idx]);
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
__global__ void chargeAssignment(float* positionX, float* positionY, int* cellX, int* cellY, float* density) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	float invArea = 1.0/cellAreaC;
	int n = npxC*npyC;
	int y = nyC;

	extern __shared__ float sdata[];	
	sdata[4*tid] 	= (idx < n) ? positionX[idx] : 0;
	sdata[4*tid+1]	= (idx < n) ? positionY[idx] : 0;
	sdata[4*tid+2] 	= (idx < n) ? cellX[idx] : 0;
	sdata[4*tid+3] 	= (idx < n) ? cellY[idx] : 0;
	__syncthreads();

	//adding to charge density of nearby points
	if (idx < n) {
		int xCell = cellX[idx];
		int yCell = cellY[idx];

		float area1 = ((positionX[idx] - cellX[idx]*dxC) * (positionY[idx] - cellY[idx]*dyC))*invArea;
		float area2 = (((cellX[idx]*dxC + dxC) - positionX[idx]) * (positionY[idx] - cellY[idx]*dyC))*invArea;
		float area3 = ((positionX[idx] - cellX[idx]*dxC) * ((cellY[idx]*dyC + dyC) - positionY[idx]))*invArea;
		float area4 = (((cellX[idx]*dxC + dxC) - positionX[idx]) * ((cellY[idx]*dyC + dyC) - positionY[idx]))*invArea;

		//way with integer logic
		density[xCell*y + yCell] 				+= (area1 * chargeC);
		density[((xCell+1)%nxC)*y + yCell] 			+= (area2 * chargeC);
		density[(xCell)*y + ((yCell+1)%nyC)] 			+= (area3 * chargeC);
		density[((xCell+1)%nxC)*y + ((yCell+1)%nyC)]		+= (area4 * chargeC);

		/*
		//using atomic add
		atomicAdd(&density[xCell*y + yCell], area1*chargeC);
		atomicAdd(&density[((xCell+1)%nxC)*y + yCell], area2*chargeC);
		atomicAdd(&density[xCell*y + ((yCell+1)%nyC)], area3*chargeC);
		atomicAdd(&density[((xCell+1)%nxC)*y + ((yCell+1)%nyC)], area4*chargeC);
		*/
		__syncthreads();
	}
}

/**
 * \brief Kernel to solve the Poisson equation in Fourier space
 *
 * \param arr	Array containing output of forward R2C Fourier transform
 * \param nyh	Physical y-dimension of Fourier transform output (not logical size due to Hermitian symmetry)
 */
__global__ void fftPoissonSolver(cufftComplex* arr, const int nyh) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int n = nxC*nyh;

	//initialising shared memory and loading array into it
	extern __shared__ float sdata[];
	sdata[2*tid] 	= (idx < n) ? arr[idx].x : 0;
	sdata[2*tid+1]	= (idx < n) ? arr[idx].y : 0;
	__syncthreads();


	float pi = 3.141592654f;

	int i, j;
	int II, JJ;
	float k1, k2;

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

		float fact = k1*k1 + k2*k2;
		float invFact = __fdividef(-1.0, fact);
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
__global__ void copyD2D(float *dest, float *src) {
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
__global__ void normaliseTransform(float *arr) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int n = nxC*nyC;

	//initialising shared memory and loading array into it
	extern __shared__ float sdata[];
	sdata[tid] = (idx < n) ? arr[idx] : 0;
	__syncthreads();

	if (idx < n) {
		float norm = __fdividef(-1.0, n);
		arr[idx] *= norm;
	}
}

/**
 * \brief Kernel to compute the electric field given the electric potential
 *
 * \param fieldX	float array to store the values of the electric field in x-direction
 * \param fieldY	float array to store the values of the electric field in y-direction
 * \param potential	float array in which the electric potential is stored
 */
__global__ void computeElectricField(float* fieldX, float* fieldY, float* potential) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int n = nxC*nyC;

	//initialising shared memory and loading array into it
	extern __shared__ float sdata[];
	sdata[tid] = (idx < n) ? potential[idx] : 0;
	__syncthreads();

	float divisorX = 2*dxC; 
	float divisorY = 2*dyC;

	if (idx < n) {
		//float i = idx%nyC;
		//float j = idx/nyC;
		float i = idx/nxC;
		float j = idx%nyC;
	
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
 * \param positionX 	float array containing x-coordinate of each particle
 * \param positionY	float array containing y-coordinate of each particle
 * \param cellX		int array containing current cell in x-direction of each particle
 * \param cellY		int array containing current cell in y-direction of each particle
 * \param fieldX	float array containing x-component of electric field at each grid point
 * \param fieldY	float array containing y-component of electric field at each grid point
 * \param forceX	float array to store x-component of force on each particle
 * \param forceY	float array to store y-component of force on each particle
 */
__global__ void forceCalculation(float* positionX, float* positionY, int* cellX, int* cellY, float* fieldX, float* fieldY, float* forceX, float* forceY) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	float invArea = 1.0/cellAreaC;
	int n = npxC*npyC;
	int y = nyC;

	//initialising shared memory and loading arrays into it
	extern __shared__ float sdata[];	
	/*
	sdata[6*tid] 	= (idx < n) ? positionX[idx] : 0;
	sdata[6*tid+1]	= (idx < n) ? positionY[idx] : 0;
	sdata[6*tid+2] 	= (idx < n) ? cellX[idx] : 0;
	sdata[6*tid+3] 	= (idx < n) ? cellY[idx] : 0;
	sdata[6*tid+4] 	= (idx < n) ? fieldX[idx] : 0;
	sdata[6*tid+5] 	= (idx < n) ? fieldY[idx] : 0;
	*/
	sdata[4*tid] 	= (idx < n) ? positionX[idx] : 0;
	sdata[4*tid+1]	= (idx < n) ? positionY[idx] : 0;
	sdata[4*tid+2] 	= (idx < n) ? cellX[idx] : 0;
	sdata[4*tid+3] 	= (idx < n) ? cellY[idx] : 0;
	__syncthreads();

	//computing force acting on each particle
	if (idx < n) {
		float tmp = 0.0;
		int xCell = cellX[idx];
		int yCell = cellY[idx];
		//printf("xCell is %d, yCell is %d\n", xCell, yCell);

		float area1 = ((positionX[idx] - xCell*dxC) * (positionY[idx] - yCell*dyC))*invArea;
		float area2 = (((xCell*dxC + dxC) - positionX[idx]) * (positionY[idx] - yCell*dyC))*invArea;
		float area3 = ((positionX[idx] - xCell*dxC) * ((yCell*dyC + dyC) - positionY[idx]))*invArea;
		float area4 = (((xCell*dxC + dxC) - positionX[idx]) * ((yCell*dyC + dyC) - positionY[idx]))*invArea;

		//computing X component of the force
		forceX[idx] = 0.0;

		//integer arithmetic
		tmp += area1 * fieldX[xCell*y + yCell];
		tmp += area2 * fieldX[((xCell+1)%nxC)*y + yCell];
		tmp += area3 * fieldX[xCell*y + ((yCell+1)%y)];
		tmp += area4 * fieldX[((xCell+1)%nxC)*y + (yCell+1)%y];
		forceX[idx] = tmp*qmC;

		//computing Y component of the force
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
 * \param positionX	float array containing x-component of position
 * \param positionY	float array containing y-component of position
 * \param velocityX	float array containing x-component of velocity
 * \param velocityY	float array containing y-component of velocity
 * \param forceX	float array containing x-component of force
 * \param forceY	float array containing y-component of force
 * \param forceOldX	float array containing x-component of force at previous time-step
 * \param forceOldY	float array containing y-component of force at previous time-step
 * \param cellX		int array containing current cell in x-direction of particle
 * \param cellY		int array containing current cell in y-dircetion of particle
 */
__global__ void positionUpdate(float* positionX, float* positionY, float* velocityX, float* velocityY, float* forceX, float* forceY, float* forceOldX, float* forceOldY, int* cellX, int* cellY) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int tid = threadIdx.x;
	int n = npC;	

	//initialising shared memory and loading arrays into it
	extern __shared__ float sdata[];	

	sdata[4*tid] 	= (idx < n) ? positionX[idx] : 0;
	sdata[4*tid+1]	= (idx < n) ? positionY[idx] : 0;
	sdata[4*tid+2] 	= (idx < n) ? velocityX[idx] : 0;
	sdata[4*tid+3] 	= (idx < n) ? velocityY[idx] : 0;
	__syncthreads();
	
	if (idx < n) {
		//compute velocity at half time step
		float vxh = velocityX[idx] + 0.5*dtC*forceOldX[idx];
		float vyh = velocityY[idx] + 0.5*dtC*forceOldY[idx];
			
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
		cellX[idx] = (int)(positionX[idx]/dxC);
		cellY[idx] = (int)(positionY[idx]/dyC);
	}
}


/**
 * \brief Function to declare device variables and call relevant kernels on the gpu
 *
 * \param 
 * \param 
 */
float picFloatGpu(float* positionX, float *positionY, float* velocityX, float* velocityY, float *xResult, float *yResult, std::shared_ptr<Params> p) {
	//cudaSetDevice(0);
	cudaSetDevice(1);

	//creating cuda streams
	cudaStream_t stream[2];
	for (int i=0; i<2; i++) {
		cudaStreamCreate(&stream[i]);
	}

	printf("starting on gpu\n");

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
	const float charge = p->electronCharge;
	const float mass = p->mass;
	const float dx = p->dx;
	const float dy = p->dy;
	const float area = p->area;
	const float qm = p->electronCharge/p->mass;
	const float dt = p->dt;

	//grid configuration for particle kernels
	dim3 dimBlock(p->blockSize);
	dim3 dimGridParticles((np/dimBlock.x)+(!(np%dimBlock.x)?0:1));

	//grid configuration for grid kernels
	int ng = nx*ny;
	dim3 dimGridCells((ng/dimBlock.x)+(!(ng%dimBlock.x)?0:1));

	//shared memory configuration
	int smem1 = dimBlock.x*sizeof(float);		//size of shared memory in each block (1 float per thread)
	int smem2 = dimBlock.x*2*sizeof(float); 	//size of shared memory in each block (2 variables per thread)
	int smem4 = dimBlock.x*4*sizeof(float); 	//size of shared memory in each block (4 variables per thread)

	//copying data to constant memory
	cudaMemcpyToSymbolAsync(xDimC, 		&xDim, 		sizeof(int), 		0,	cudaMemcpyHostToDevice, 	stream[0]);
	cudaMemcpyToSymbolAsync(yDimC, 		&yDim, 		sizeof(int),		0,	cudaMemcpyHostToDevice, 	stream[1]);
	cudaMemcpyToSymbolAsync(nxC, 		&nx, 		sizeof(int),		0,	cudaMemcpyHostToDevice, 	stream[0]);
	cudaMemcpyToSymbolAsync(nyC, 		&ny, 		sizeof(int),		0,	cudaMemcpyHostToDevice, 	stream[1]);
	cudaMemcpyToSymbolAsync(npxC, 		&npx,		sizeof(int),		0, 	cudaMemcpyHostToDevice,		stream[0]);
	cudaMemcpyToSymbolAsync(npyC, 		&npy,	 	sizeof(int),		0,	cudaMemcpyHostToDevice, 	stream[1]);
	cudaMemcpyToSymbolAsync(npC, 		&np,	 	sizeof(int),		0,	cudaMemcpyHostToDevice, 	stream[0]);
	cudaMemcpyToSymbolAsync(chargeC, 	&charge,	sizeof(float),		0,	cudaMemcpyHostToDevice, 	stream[1]);
	cudaMemcpyToSymbolAsync(massC, 		&mass,		sizeof(float),		0,	cudaMemcpyHostToDevice,		stream[0]);
	cudaMemcpyToSymbolAsync(dxC, 		&dx, 		sizeof(float), 		0, 	cudaMemcpyHostToDevice, 	stream[1]);
	cudaMemcpyToSymbolAsync(dyC, 		&dy, 		sizeof(float),		0, 	cudaMemcpyHostToDevice, 	stream[0]); 
	cudaMemcpyToSymbolAsync(cellAreaC, 	&area, 		sizeof(float),		0, 	cudaMemcpyHostToDevice, 	stream[1]); 
	cudaMemcpyToSymbolAsync(qmC,	 	&qm, 		sizeof(float),		0, 	cudaMemcpyHostToDevice, 	stream[0]); 
	cudaMemcpyToSymbolAsync(dtC,	 	&dt, 		sizeof(float),		0, 	cudaMemcpyHostToDevice, 	stream[1]); 

	//pointers to device memory
	float *positionXGpu, *positionYGpu;
	int *currentCellXGpu, *currentCellYGpu;
	float *velocityXGpu, *velocityYGpu;
	float *forceXGpu, *forceYGpu;
	float *forceXOldGpu, *forceYOldGpu;
	float *chargeDensityGpu;
	float *electricPotentialGpu;
	float *electricFieldXGpu, *electricFieldYGpu;

	//allocating device memory for data arrays
	cudaMalloc((void**) &positionXGpu,		np*sizeof(float)); 	//memory for particle positions
	cudaMalloc((void**) &positionYGpu, 		np*sizeof(float)); 	//memory for particle positions
	cudaMalloc((void**) &velocityXGpu, 		np*sizeof(float)); 	//memory for velocity of particles
	cudaMalloc((void**) &velocityYGpu,		np*sizeof(float)); 	//memory for velocity of particles
	cudaMalloc((void**) &currentCellXGpu, 		np*sizeof(int)); 	//memory for current cell of each particle
	cudaMalloc((void**) &currentCellYGpu, 		np*sizeof(int)); 	//memory for current cell of each particle
	cudaMalloc((void**) &forceXGpu, 		np*sizeof(float)); 	//memory for force acting on particles
	cudaMalloc((void**) &forceYGpu, 		np*sizeof(float)); 	//memory for force acting on particles
	cudaMalloc((void**) &forceXOldGpu, 		np*sizeof(float)); 	//memory for force acting on particles at previous timestep
	cudaMalloc((void**) &forceYOldGpu, 		np*sizeof(float)); 	//memory for force acting on particles at previous timestep
	cudaMalloc((void**) &chargeDensityGpu, 		nx*ny*sizeof(float)); 	//memory for charge density at all grid points
	cudaMalloc((void**) &electricPotentialGpu, 	nx*ny*sizeof(float)); 	//memory for electric potential at all grid points
	cudaMalloc((void**) &electricFieldXGpu, 	nx*ny*sizeof(float)); 	//memory for electric field at all grid points
	cudaMalloc((void**) &electricFieldYGpu, 	nx*ny*sizeof(float)); 	//memory for electric field at all grid points

	//copying particle data from host	
	cudaMemcpyAsync(positionXGpu,	positionX, 	np*sizeof(float), cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(positionYGpu,	positionY, 	np*sizeof(float), cudaMemcpyHostToDevice, stream[1]);
	cudaMemcpyAsync(velocityXGpu,	velocityX, 	np*sizeof(float), cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(velocityYGpu, 	velocityY,	np*sizeof(float), cudaMemcpyHostToDevice, stream[1]);

	//creating cuFFT plans
	cufftHandle planForward, planInverse;
	cufftCreate(&planForward);
	cufftCreate(&planInverse);
  	cufftPlan2d(&planForward, nx, ny, CUFFT_R2C);
	cufftPlan2d(&planInverse, nx, ny, CUFFT_C2R);
	
	//allocating memory for fft
	const int nyh = ny/2 + 1;
	float *in;
	cufftComplex *out;
	cudaMalloc((void**) &out, nx*nyh*sizeof(cufftComplex));

	//starting iteration
	float t = 0;
	int iter = 1;
	printf("Starting iteration on GPU.\n");
	while (t < p->tmax) {
		//setting current cells of each particle
		currentCell<<<dimGridParticles,dimBlock,smem2,stream[0]>>>(positionXGpu, positionYGpu, currentCellXGpu, currentCellYGpu);	

		//interpolating charge to gridpoints
		chargeAssignment<<<dimGridParticles,dimBlock,smem4,stream[0]>>>(positionXGpu, positionYGpu, currentCellXGpu, currentCellYGpu, chargeDensityGpu);

		//forward R2C transform
		in = chargeDensityGpu;
		cufftExecR2C(planForward, in, out);

		//computing poisson equation in Fourier space
		fftPoissonSolver<<<dimGridCells,dimBlock,smem2,stream[0]>>>(out, nyh);
		
		//inverse C2R transform and normalisation
		cufftExecC2R(planInverse, out, in);
		normaliseTransform<<<dimGridCells,dimBlock,smem2,stream[0]>>>(in);

		//copy FFT Poisson output to electric potential array
		copyD2D<<<dimGridCells,dimBlock,0,stream[0]>>>(electricPotentialGpu, in);

		//computing electric field from electric potential
		computeElectricField<<<dimGridCells,dimBlock,smem1,stream[0]>>>(electricFieldXGpu, electricFieldYGpu, electricPotentialGpu);

		//computing force on each particle
		forceCalculation<<<dimGridParticles,dimBlock,smem4,stream[0]>>>(positionXGpu, positionYGpu, currentCellXGpu, currentCellYGpu, electricFieldXGpu, electricFieldYGpu, forceXGpu, forceYGpu);

		//updating particle positions
		positionUpdate<<<dimGridParticles,dimBlock,smem4,stream[0]>>>
			(positionXGpu, positionYGpu, velocityXGpu, velocityYGpu, forceXGpu, forceYGpu, forceXOldGpu, forceYOldGpu, currentCellXGpu, currentCellYGpu);

		//copy new force to forceOld arrays
		copyD2D<<<dimGridCells,dimBlock,0,stream[0]>>>(forceXOldGpu, forceXGpu);
		copyD2D<<<dimGridCells,dimBlock,0,stream[1]>>>(forceYOldGpu, forceYGpu);

		iter++;

		t += dt;
	}
	printf("GPU finished computation\n");

	//copying results back to the host
	cudaMemcpyAsync(xResult, positionXGpu, np*sizeof(float), cudaMemcpyDeviceToHost, stream[0]);
	cudaMemcpyAsync(yResult, positionYGpu, np*sizeof(float), cudaMemcpyDeviceToHost, stream[1]);
	
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

	printf("finished on gpu\n");

	//return elapsed time in ms
	return (elapsedTime/1000);
}

