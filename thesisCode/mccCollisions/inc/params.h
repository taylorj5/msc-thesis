/**
 * \file params.h
 * \brief Header file containing some structures for the Particle objects
 * \author J. Taylor
 * \version 1.0
 * \date 2020-06-05
 */

#ifndef PARAMS_H_BLZU98JT
#define PARAMS_H_BLZU98JT

#include <random>
#include <fstream>
#include <complex>

//extern std::default_random_engine gen;

/**
 * \brief Structure to store parameters needed by program
 */
struct Params {
	int blockSize = 64;

	/* simulation variables */
	int xDim 		= 10; 		//!< domain of the solution: 0 <= x <= L (Debye lengths)
	int yDim 		= 10;		//!< domain of the solution in y direction
	int nx 			= 10;		//!< number of cells in x direction
	int ny 			= 10;		//!< number of cells in y direction
	int numParticlesX	= 10;		//!< number of particles in x direction
	int numParticlesY	= 10;		//!< number of particles in y direction
	int numParticles;			//!< total number of particles in simulation
	double dt		= 1e-7;		//!< time step (in 1/plasma frequency)
	double tmax 		= 1e-4;		//!< simulation runs from t0 to tMax

	/* particle properties */
	double electronCharge 		= -100;			//!< charge of a superparticle in units of e
	double mass			= 100;			//!< mass of a superparticle
	double ax	 		= 0.912871;		//!< smoothed particle size in x direction
	double ay 			= 0.912871;		//!< smoothed particle size in y direction
	double vtx 			= 1.0;			//!< x component of thermal velocity
	double vty 			= 1.0;			//!< y component of thermal velocity
	double vdx 			= 0.0;			//!< x component of drift velocity
	double vdy 			= 0.0;			//!< y component of drift velocity
	
	double dx;			// !<x increment
	double dy;			// !<y increment
	double area;			// !<area of one cell

	/* RNG seed */
	long int seed;
};

/**
 * \brief Simple structure to store coordinates of a point on the grid
 */
struct Point {
	double x;		//!< x-coordinate
	double y;		//!< y-coordinate
};

/**
 * \brief Simple structure to store coordinates of a complex number on the grid
 */
struct complexPoint {
	std::complex<double> x;		//!< x-coordinate
	std::complex<double> y;		//!< y-coordinate
};

/**
 * \brief Overloading output stream operator for a Point. Simply prints x and y coordinates of Point
 *
 * \param os 	Reference to output stream
 * \param in 	Reference to the Point we wish to print
 *
 * \return 		Reference to output stream
 */
inline std::ostream & operator<< (std::ostream & os, Point & in) {
	os << in.x << ',' << in.y;
	return os;
}

#endif /* end of include guard: PARAMS_H_BLZU98JT */
