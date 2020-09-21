/**
 * \file person.cc
 * \brief Contains class functions and operators for the Particle class
 * \author J. Taylor
 * \version	1.0
 * \date 2020-06-05
 */

#include <iostream>
#include "particle.h"

/**
 * \brief Defining the constructor for the Particle class
 *
 * It sets the orig_location to the input Point and sets the params pointer to the Params struct.
 * Also sets the location struct to the input Point s 
 *
 * \params s	Point struct containing coordinates we wish to add to the particle
 * \param p 	Shared pointer to Params struct containing simulation parameters
 */
Particle::Particle(Point s, std::shared_ptr<Params> p) : originalPosition {s}, params {p} {
	position = originalPosition;
	currentCell.x = (int)(position.x/p->dx);
	currentCell.y = (int)(position.y/p->dy);
	force.x = 0.0;
	force.y = 0.0;
	forceOld.x = force.x;
	forceOld.y = force.y;
};	

/** 
 * \brief Function to compute the total force on each particle from nearby grid points
 *
 * \param grid	Grid object from which we interpolate the force value
 * \param p		smart pointer to Params struct which contains simulation variables
 */
void Particle::forceCalculation(Grid& grid, std::shared_ptr<Params> p) {
	//calculating percentage of electron's effective charge goes to each vertex
	double totalArea = p->area;
	double area1 = ((position.x - currentCell.x*p->dx) * (position.y - currentCell.y*p->dy))/totalArea;
	double area2 = (((currentCell.x*p->dx + p->dx) - position.x) * (position.y - currentCell.y*p->dy))/totalArea; 
	double area3 = ((position.x - currentCell.x*p->dx) * ((currentCell.y*p->dy + p->dy) - position.y))/totalArea;
	double area4 = (((currentCell.x*p->dx + p->dx) - position.x) * ((currentCell.y*p->dy + p->dy) - position.y))/totalArea;

	double qm = p->electronCharge/p->mass;

	force.x = 0.0;
	force.x += area1 * grid.electricField[currentCell.x*grid.y + currentCell.y].x;
	force.x += area2 * grid.electricField[((int)(currentCell.x + 1)%p->nx)*grid.y + currentCell.y].x;
	force.x += area3 * grid.electricField[currentCell.x*grid.y + (int)(currentCell.y + 1)%p->ny].x;
	force.x += area4 * grid.electricField[((int)(currentCell.x + 1)%p->nx)*grid.y + (int)(currentCell.y + 1)%p->ny].x;
	force.x *= qm;

	force.y = 0.0;
	force.y += area1 * grid.electricField[currentCell.x*grid.y + currentCell.y].y;
	force.y += area2 * grid.electricField[((int)(currentCell.x + 1)%p->nx)*grid.y + currentCell.y].y;
	force.y += area3 * grid.electricField[currentCell.x*grid.y + (int)(currentCell.y + 1)%p->ny].y;
	force.y += area4 * grid.electricField[((int)(currentCell.x + 1)%p->nx)*grid.y + (int)(currentCell.y + 1)%p->ny].y;
	force.y *= qm;	
}

/** 
 * \brief Function to solve the equations of motion for each particle
 * Uses the ``kick-drift-kick'' implementation of the Leapfrog method
 *
 * \param p		Smart pointer to Params struct which contains simulation variables
 */
void Particle::positionUpdate(std::shared_ptr<Params> p) {
	double dt = p->dt;

	//compute velocity at half time step
	double vxh = velocity.x + 0.5*dt*forceOld.x;
	double vyh = velocity.y + 0.5*dt*forceOld.y;

	//update position
	position.x += vxh*dt;
	position.y += vyh*dt;

	//if (position.x >= p->xDim) {
	if (position.x > p->xDim) {
		position.x -= p->xDim;
	}
	else if (position.x < 0) {
		position.x += p->xDim;
	}

	//if (position.y >= p->yDim) {
	if (position.y > p->yDim) {
		position.y -= p->yDim;
	}
	else if (position.y < 0) {
		position.y += p->yDim;
	}

	//update velocity
	velocity.x = vxh + 0.5*dt*force.x;
	velocity.y = vyh + 0.5*dt*force.y;

	//update old force
	forceOld = force;

	//update current cell
	currentCell.x = (int)(position.x/p->dx);
	currentCell.y = (int)(position.y/p->dy);
}

/** 
 * \brief Function to simulate collisions with ions
 *
 * \param p		Smart pointer to Params struct which contains simulation variables
 */
void Particle::collision(std::mt19937& gen) {
	//define collision probability
	double prob = 0.8;

	//generate random number, r
	std::uniform_real_distribution<double> dist{0.0, 1.0};
	double r = dist(gen);	

	//if probability > random number, collision occurs
	if (prob > r) {
		double max1 = 2*(dist(gen) + dist(gen) + dist(gen) - 1.5);
		double max2 = 2*(dist(gen) + dist(gen) + dist(gen) - 1.5);
		velocity.x = max1;
		velocity.y = max2;
	}
}

/** 
 * \brief Function to set the private drift velocity variable for a particle
 *
 * \param v 	Point containing drift velocity vector we wish to add to the particle
 */
void Particle::setVelocity(Point v) {
	velocity = v;
}

/** 
 * \brief Function to get the private drift velocity variable for a particle
 */
Point Particle::getVelocity() {
	return velocity;
}

/** 
 * \brief Function to get the private drift velocity variable for a particle
 */
Point Particle::getPosition() {
	return position;
}

/**
 * \brief Overloading the output stream operator for the Particle class.
 * 	
 * 	It will print the x and y coordinates stored in the Point struct of the Particle
 *
 * 	\param os 	Reference to output stream
 * 	\param in 	Reference to Particle whos coordinates we wish to print
 *
 * 	\return 	Reference to output stream
 */
std::ostream& operator<< (std::ostream &os, Particle &in) {
	os << in.position; 			//'\n' over endl for performance
	return os;
}
