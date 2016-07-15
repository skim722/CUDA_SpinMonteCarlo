/*****************************************************/
/* crystal.h -- Class definitions for crystal.       */
/*              (c) J.B. Sturgeon, October, 24, 2006 */
/* modified MPSurh 10/26/06-11/03/06		     */
/* modified MPSurh 6/27/14 - allow multiple species  */
/*****************************************************/
#ifndef CRYSTAL_H
#define CRYSTAL_H
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>

static const int LATTICED=3; // lattice dimension
                             // This MUST BE 3!

using namespace std;


class Crystal {
public:
  Crystal();                        //Default constructor
  Crystal(const string);            //constructor from input file
  Crystal(const Crystal & ref);     //Copy constructor
  Crystal(int, int, int, const Crystal &);    //Copy constructor, replicated
  ~Crystal();                       //Deletes memory allocated to member basis

  const Crystal &operator =(const Crystal &);  //Assignment operator
  bool operator ==(const Crystal &); //Equality operator
  bool operator !=(const Crystal &); //Inequality operator

  int setBasisToZero();

  int getNumSpecies() const { return nSpecies; }

//  int setNumAtoms(int i, int n);
  int getNumAtoms(int i) const { return nAtoms[i]; }
  string getSpeciesName(int i) const { return nameSpecies[i]; }
  
  int getIJthBasis(int iSp, int jAt, double vecg[LATTICED]) const
      {for (int ig=0;ig<LATTICED;ig++) vecg[ig] = basis[iSp][jAt][ig];}

  int getIthLattice(int ith, double vecg[LATTICED]) const
      {
       if (ith >=LATTICED) {cout << "bad lattice direction!\n"; exit(1);}
       for (int ig=0;ig<LATTICED;ig++) vecg[ig] = lattice[ith][ig];
      }  

  int getIthPrimLattice(int ith, double vecg[LATTICED]) const
      {
       if (ith >=LATTICED) {cout << "bad lattice direction!\n"; exit(1);}
       for (int ig=0;ig<LATTICED;ig++) vecg[ig] = primLattice[ith][ig];
      }  

private:
  int nSpecies;                       // number of species
  string* nameSpecies;                // name of species
  int * nAtoms;                       // number of basis atoms
  double ***basis;                    // basis atom coordinates as nAtoms vectors of size LATTICED
  double lattice[LATTICED][LATTICED]; // LATTICED lattice vectors, each of size LATTICED
  double primLattice[LATTICED][LATTICED]; // primitive lattice vectors

  int deallocateMemory(); //Releases memory used by basis - also used if/when re-sizing

  int setnSpecies(int nSp)
                 {
                  if (basis != NULL) { for (int i=0;i<nSpecies;i++)
                                          {
                                           for (int j=0;j<nAtoms[i];j++)
                                                delete [] basis[i][j];
                                           delete basis[i];
                                     }    }

                  nSpecies = nSp;
                  if(nAtoms != NULL) { delete [] nAtoms;}
                  else               { nAtoms = new int [nSpecies];}

                  basis = new double**[nSpecies];
                  return 0;
                 }

};

#endif //CRYSTAL_H
