/********************************************************/
/* magnet.h -- Class definitions for magnet.            */
/* Supplement the unit cell lattice and atomic basis    */
/* from Crystal class with a derived neighbor list.     */
/* Or alternatively input neighbor information directly */
/* with no associated crystallographic information.     */
/********************************************************/
// Each atom in the simulation is indexed 1-NTOT,
// this list is associated with global arrays, gl***.
// 
#ifndef MAGNET_H
#define MAGNET_H
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include "crystal.h"

static const int DIMENSIONS=3; // spin dimension
                               // This can be varied.

using namespace std;

class Magnet: public Crystal {
public:
  Magnet();                               //Default constructor
  Magnet(const string m, const string c); //constructor from Crystal input file
  Magnet(const Magnet & ref);             //Copy constructor
  Magnet(int n, const Magnet & ref);      //Copy constructor, replicated
  ~Magnet();                              //Deletes memory allocated 

  const Magnet &operator =(const Magnet &);  //Assignment operator
  bool operator ==(const Magnet &);          //Equality operator
  bool operator !=(const Magnet &);          //Inequality operator

  double Energy(int ith);   //Heisenberg energy of ith spin
  double totEnergy();       //Heisenberg energy of entire cell

  double * getIthSpin(int ith) {return glSpins[ith];}
private:

// Inherit crystallographic data.
Crystal xtlData;

// Reformat Crystal data:
// All nTot atoms are grouped into a global 1D list.
// ith global atom is indexed in Crystal as 
// glAtomID[i]'th member of glSpeciesID[i].
int nTot;
int*   glSpeciesID;
int*   glAtomID;

// Each site has a spin vector.
double** glSpins;          // ith = 1 to nTot

// canned procedure to convert Crystal to global variables.
int reformatCrystal();

// canned procedure to compute neighbor lists out to rmax
int makeNbrLists(double rmax);

// the following has not been implemented yet.
// Define neighbor shells and neighbor lists.
// If this data is input directly, then probably
// nShells=1 and shellRadius=undefined;
                         // otherwise:
int*     nShells;        // # of shells around ith site
int**    shellSize;      // # of neighbors in each such shell
double** shellRadius;    // nbr distance of each shell
int***   shellNbr;       // index list (1-nTot) of atoms in each shell

// Define Heisenberg coupling constants.
double*** J;

};

#endif //MAGNET_H
