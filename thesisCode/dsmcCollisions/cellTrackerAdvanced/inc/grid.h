/**
 * \file grid.h
 * \brief Header file to store Grid class definition
 * \author J. Taylor
 * \version 1.0
 * \date 2020-06-05
 */

#ifndef GRID_H_CXHP01LT
#define GRID_H_CXHP01LT

#include <ostream>
#include <vector>
#include <complex>

//including particle.h if it hasn't already been included
#include "particle.h"
class Particle;

class Grid {
public:
	Grid(int width, int height, int z, std::shared_ptr<Params> p); 		//constructor

	void chargeAssignment(const std::vector<Particle> &particles);
	void computeElectricField();
	void resetGrid();
	int x; 			//!<size of grid in x-direction
	int y;			//!<size of grid in y-direction

	std::vector<double> chargeDensity;
	std::vector<double> electricPotential;
	std::vector<Point> electricField;
	std::vector< std::vector<int> > cellTracker;

	friend std::ostream& operator<< (std::ostream & os, Grid & in);

private:
	int size;		//!<total size of grid
	int population;		//!<original population of grid
	std::shared_ptr<Params> p;

	friend class Particle;	
};

#endif //end of include guard: GRID_H_CXHP01LT
