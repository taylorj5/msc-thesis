#include <iostream>
#include <stdio.h>
#include <complex>

#include <fftw3.h>
#include "fourier.h"
#include "grid.h"

/**
 * \brief Function to calculate tables needed for a 2D real to complex FFT and its inverse
 *
 * \param grid 	Reference to grid object containing the arrays we wish to modify
 * \param p		Smart pointer to Params object containing the necessary constants
 */
void fftPoissonSolver(Grid& grid, std::shared_ptr<Params> p) {
	const double pi = M_PI;
	const int nx = p->nx;			//size of domain in x direction
	const int ny = p->ny;			//size of domain in y direction
	const int nyh = (p->ny)/2 + 1;		//half the y domain, padded as output is Hermitian symmetric
	const double norm = 1.0/(p->nx*p->ny);	//normalisation factor for output

	//declaring fft components
	double* in;
	fftw_complex *out;
	in = (double*)fftw_malloc(sizeof(double)*nx*ny);
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*nx*nyh);

	//filling input array with charge densities
	for (auto i=0; i<nx; i++) {
		for (auto j=0; j<ny; j++) {
			in[i*ny+j] = grid.chargeDensity[(i+1)*grid.y + (j+1)];
		}
	}

	//creating execution plans for the forward and inverse Fourier transforms
	fftw_plan fwrd = fftw_plan_dft_r2c_2d(nx, ny, in, out, FFTW_MEASURE);
	fftw_plan bwrd = fftw_plan_dft_c2r_2d(nx, ny, out, in, FFTW_MEASURE);

	//executing forward fft
	fftw_execute(fwrd);

	//solve the poisson equation to obtain the electric potential in frequency space
	int II, JJ;
	double k1, k2;

	for (int i=0; i<nx; i++) {
		if (2*i < nx) {
			II = i;
		}
		else {
			II = i - nx;
		}	
		k1 = 2*pi*II;

		for (int j=0; j<nyh; j++) {
			if (2*j < nyh) {
				JJ = j;
			}
			else {
				JJ = j - ny;
			}
			k2 = 2*pi*JJ;
			//nabla^2 phi = -rho/epsilon
			//in fourier space, phi = -rho/(-k^2*epsilon) = rho/(epsilon*k^2)
			double fact = k1*k1 + k2*k2;
			if (fabs(fact) < 1e-14) { //avoids division by zero
				out[i*nyh+j][0] = 0.0;
				out[i*nyh+j][1] = 0.0;
			} else {
				out[i*nyh+j][0] /= -fact;
				out[i*nyh+j][1] /= -fact;
			}
		}
	}


	//perform inverse fft to obtain potential in real space
	fftw_execute(bwrd);

	//normalising the output of the inverse fft
	for (auto i=0; i<nx; i++) {
		for (auto j=0; j<ny; j++) {
			in[i*ny+j] *= norm;
			in[i*ny+j] *= norm;
		}
	}
	
	int nxe = p->nx+2;
	int nye = p->ny+2;
	//the output of the inverse FFT is the electric potential phi
	//as FFTW does not normalise the output, we must rescale the output by the size of the system 
	for (auto i=1; i<nxe-1; i++) {
		for (auto j=1; j<nye-1; j++) {
			//grid.electricPotential[i][j] = out[(i-1)*ny+(j-1)][1];
			//grid.electricPotential[i*grid.y+j] = mem[(i-1)*ny+(j-1)][0];
			grid.electricPotential[i*grid.y+j] = in[(i-1)*ny+(j-1)];
		}
	}	

	//setting ghost cells to have periodic boundary conditions
	for (auto i=1; i<nye-1; i++) {
		grid.electricPotential[i] = grid.electricPotential[p->nx*grid.y + i];		
		grid.electricPotential[(nxe-1)*grid.y+i] = grid.electricPotential[grid.y + i];
	}
	for (auto i=1; i<nxe-1; i++) {
		grid.electricPotential[i*grid.y] = grid.electricPotential[i*grid.y + p->ny];
		grid.electricPotential[i*grid.y + (nye-1)] = grid.electricPotential[i*grid.y + 1];
	}

	//deallocate both the fft and inverse fft plans
	fftw_destroy_plan(fwrd);
	fftw_destroy_plan(bwrd);

	//will remove all associated data incl. all accumulated wisdom and available algorithms
	//fftwf_cleanup();
	free(in);

	return;
}
