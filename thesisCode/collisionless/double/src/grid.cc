/**
 * \file grid.cc
 * \brief Contains class functions and operators for the Grid class
 * \author J. Taylor 
 * \version 1.0
 * \date 2020-06-05
 */

#include "grid.h"
#include "params.h"
#include "particle.h"

#include <iostream>

/**
 * \brief Constructor for the Grid class
 */
Grid::Grid(const int width, const int height, const int pop, std::shared_ptr<Params> params) : x {width+2}, y {height+2}, population {pop}, p{params} {
	//setting simulation parameters
	size = x*y;
	Point zero {0.0, 0.0};
	chargeDensity.resize(size, 0.0);
	electricPotential.resize(size, 0.0);
	electricField.resize(size, zero);
};

/**
 * \brief Class function to reset the Grid values
 */
void Grid::resetGrid() {
	for (auto i=0; i<x; ++i) {
		for (auto j=0; j<y; ++j) {
			chargeDensity[i*y+j] = 0.0;
		}
	}
	return;
};

/**
 * \brief Function to distribute the charges of electrons to the fixed grid points w/ periodic boundaary conditions
 * It uses an area-based distribution to divide up the effective charge of each electron
 *
 * \param particles 	An std::vector containing all the particle objects
 * \param p				A smart pointer to a struct containing the required params
 */
void Grid::chargeAssignment(const std::vector<Particle> &particles) {
	resetGrid();
	//finding charge distribution across the grid
	for (std::vector<Particle>::const_iterator it = particles.begin(); it!=particles.end(); ++it) {
		//calculating percentage of electron's effective charge goes to each vertex
		double totalArea = p->area; 
		double area1 = (it->position.x - it->currentCell.x*p->dx) * (it->position.y - it->currentCell.y*p->dy)/totalArea;
		double area2 = ((it->currentCell.x*p->dx + p->dx) - it->position.x) * (it->position.y - it->currentCell.y*p->dy)/totalArea; 
		double area3 = (it->position.x - it->currentCell.x*p->dx) * ((it->currentCell.y*p->dy + p->dy) - it->position.y)/totalArea;
		double area4 = ((it->currentCell.x*p->dx + p->dx) - it->position.x) * ((it->currentCell.y*p->dy + p->dy) - it->position.y)/totalArea;
		//std::cout << area1 << ',' << area2 << ',' << area3 << ',' << area4 << '\n';

		//distributing effective charge of electron to each cell vertex
		chargeDensity[(1+it->currentCell.x)*y + (1+it->currentCell.y)]						+= (area1 * p->electronCharge);
		chargeDensity[(1+(int)(it->currentCell.x + 1)%p->nx)*y + (1+(int)it->currentCell.y)] 			+= (area2 * p->electronCharge);
		chargeDensity[(1+(int)it->currentCell.x)*y + (1+(int)(it->currentCell.y + 1)%p->ny)] 			+= (area3 * p->electronCharge);
		chargeDensity[(1+(int)(it->currentCell.x + 1)%p->nx)*y + (1+(int)(it->currentCell.y + 1)%p->ny)]	+= (area4 * p->electronCharge);
	}

	//setting ghost cells to have periodic boundary conditions
	int nxe = p->nx+2;
	int nye = p->ny+2;
	for (auto i=0; i<nye; i++) {
		chargeDensity[i] = chargeDensity[p->nx*y + i];		
		chargeDensity[(nxe-1)*y + i] = chargeDensity[y+i];
	}
	for (auto i=1; i<nxe-1; i++) {
		chargeDensity[i*y] = chargeDensity[i*y+p->ny];
		chargeDensity[i*y+(nye-1)] = chargeDensity[i*y+1];
	}
};


/**
 * \brief Function to compute the electric field at each grid point, given the electric potential
 * It uses a simple finite difference equation in each direction to find the vector value
 *
 * \param p	A smart pointer to a struct containing the required params
 */
void Grid::computeElectricField() {
	int nxe = p->nx+2;
	int nye = p->ny+2;
	for (auto i=1; i<nxe-1; i++) {
		for (auto j=1; j<nye-1; j++) {
			//electricField[i*y+j].x = (electricPotential[(i-1)*y+j] - electricPotential[(i+1)*y+j])/(2*p->dx);
			//electricField[i*y+j].y = (electricPotential[i*y+(j-1)] - electricPotential[i*y+(j+1)])/(2*p->dy);
			electricField[i*y+j].x = (electricPotential[i*y+(j-1)] - electricPotential[i*y+(j+1)])/(2*p->dx);
			electricField[i*y+j].y = (electricPotential[(i-1)*y+j] - electricPotential[(i+1)*y+j])/(2*p->dy);
		}
	}
}
