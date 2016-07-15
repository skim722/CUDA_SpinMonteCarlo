/********************************************************/
/* magnet.h -- Class definitions for magnet.            */
/* Supplement the unit cell (lattice and atomic basis)  */
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
#include "vector.h"
#include "KKR.h"

static const int DIMENSIONS=3; // spin dimension
                               // This can be varied.

using namespace std;

class Magnet: public Crystal {
public:
  Magnet();                               //Default constructor
  Magnet(const string m, const string c); //constructor from Crystal input file
  Magnet(const Magnet & ref);             //Copy constructor
  Magnet(int n1, int n2, int n3, const Magnet & ref);      //Copy constructor, replicated
  ~Magnet();                              //Deletes memory allocated 

  const Magnet &operator =(const Magnet &);  //Assignment operator
  bool operator ==(const Magnet &);          //Equality operator
  bool operator !=(const Magnet &);          //Inequality operator

  double Energy(int ith);   //Heisenberg energy of ith spin
  double totEnergy();       //Heisenberg energy of entire cell

  double getIthSpin(int ith, double v[DIMENSIONS]) const {for (int i=0;i<DIMENSIONS;i++)v[i]=glSpins[ith][i];}
  double writeIthSpin(int ith, double v[DIMENSIONS]){for (int i=0; i<DIMENSIONS; i++) glSpins[ith][i]=v[i];}
  ///
  int getnTot() const {return nTot;}
  int getIJthShellSize(int ith, int jth) const {return shellSize[ith][jth];}
  int getIJKthNbr(int ith, int jth, int kth) const {return shellNbr[ith][jth][kth]; }
  double getIJKthJ(int ith, int jth, int kth) const {return J[ith][jth][kth]; }
  string getglName(int ith)const {return glName[ith];}
   ///
  double getRmax() const {return kkr.getRmax();}
  double setRmax() {rmax = kkr.getRmax();}
  int getIthLattice(int ith, double vecg[LATTICED]) const {xtlData.getIthLattice(ith,vecg);}
  int secondPass(int nTot, int nLines, int glSpeciesID[], int glAtomID[],
                 int* nShells, double** shellRadius2,
                 int** shellSize, int*** shellNbr, double rmax);
int getIthNShell(int i) const {return nShells[i];}
double setT(double Tin) {T=Tin;}
double getT() const {return T;}
double setH(double h[3]) {H[0]=h[0]; H[1]=h[1]; H[2]=h[2];}
double getH(double h[3]) const {h[0]=H[0]; h[2]=H[1]; h[2]=H[2];}
int getnMonte() const {return nMonte;}
int setnMonte(int i) {nMonte=i;}

private:

// Inherit crystallographic data.
Crystal xtlData;

// Reformat Crystal data:
// All nTot atoms are grouped into a global 1D list.
// ith global atom is indexed in Crystal as 
// glAtomID[i]'th member of glSpeciesID[i].
int nTot;
int*     glSpeciesID;
string*  glName;
int*     glAtomID;
 double** glCoord; //
double rmax;  // this is approx the cutoff for Heisenberg interactions

// Each site has a spin vector.
double** glSpins;          // ith = 1 to nTot

// canned procedure to convert Crystal to global variables.
int reformatCrystal();

// canned procedure to compute neighbor lists out to rmax
int makeNbrLists(double rmax);
int statusReport();

// Define neighbor shells and neighbor lists.
// If this data is input directly, then it is possible to have
// nShells=1 and shellRadius2=undefined;
                         // otherwise:
int*     nShells;        // # of shells around ith site
int**    shellSize;      // # of neighbors in each such shell, sitesXshells
double** shellRadius2;   // squared nbr distance of each shell, sitesXshells
int***   shellNbr;       // index of atoms in each shell, sitesXshellsXnbrs

// Define Heisenberg coupling constants over sitesXshellsXnbrs
double*** J;

// save temperature as one of the magnet parameters.
double T;

// save external field
double H[3];

// save step count
int nMonte;

KKR kkr;

};

#endif //MAGNET_H
