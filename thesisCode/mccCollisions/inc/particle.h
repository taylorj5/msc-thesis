/**
 * \file particle.h
 * \brief Header file containing class definition for Particle class
 * \author J. Taylor
 * \version 1.0
 * \date 2020-06-05
 */

#ifndef PARTICLE_H_UHGN67UH
#define PARTICLE_H_UHGN67UH

#include <memory>
#include "params.h"

#include "grid.h"
class Grid;

/**
 * \brief Class to store a Particle object for use in the simulation
 */
class Particle {
public:
	Particle (Point s, std::shared_ptr<Params> p);

	void setVelocity(Point v);
	Point getVelocity();
	Point getPosition();

	Point position;
	Point currentCell;

	void forceCalculation(Grid& grid, std::shared_ptr<Params> p);
	void positionUpdate(std::shared_ptr<Params> p);
	void collision(std::mt19937& gen);

	friend std::ostream& operator<< (std::ostream &os, Particle &in);

private:
	Point force;
	Point forceOld;
	Point originalPosition;
	Point velocity;
	std::shared_ptr<Params> params;

	friend class Grid;
};

#endif //end of include guard : PARTICLE_H_UHGN67UH
