/*****************************************************/
/* magnet.h -- Class definitions for magnet.       */
/*****************************************************/
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
#include <assert.h>
#include "magnet.h"

using namespace std;

int findnRhomb(double rmax, double lattice[3][3], int nRhomb[3]);

//**********************************************************************
// Default constructor inherits default Crystal, leaves arrays undefined
Magnet::Magnet() : Crystal()
{
 nTot      = 0;
 glSpeciesID = NULL;
 glAtomID    = NULL;
 glSpins     = NULL;
 xtlData=Crystal();
}

//**********************************************************************
// Constructor from 2 input files, including crystal data.
Magnet::Magnet(string cName, string mName) : Crystal(cName)
{
 nTot      = 0;
 glSpeciesID = NULL;
 glAtomID    = NULL;
 glSpins     = NULL;

 xtlData=Crystal(cName);

 // convert xtlData to "global" format.
 reformatCrystal();

 ifstream fin;
 fin.open(mName.c_str());

 double dum;
 for (int iTo=0;iTo<nTot;iTo++)
     {fin >> glSpins[iTo][0];
      fin >> glSpins[iTo][1];
      fin >> glSpins[iTo][2];}

 fin.close();

}

//**********************************************************************
// Copy constructor.
Magnet::Magnet(const Magnet &refMagnet) : Crystal(refMagnet)
{
 nTot      = 0;
 glSpeciesID = NULL;
 glAtomID    = NULL;
 glSpins     = NULL;
 // copy Crystal data
 xtlData=refMagnet.xtlData;

 // directly copy all global data
 nTot = refMagnet.nTot;

 if (nTot>0)
    {
     glSpeciesID = new int [nTot];
     glAtomID    = new int [nTot];
     glSpins     = new double* [nTot];
     for (int iTo=0;iTo<nTot;iTo)
         {
          glSpeciesID[iTo] = refMagnet.glSpeciesID[iTo];
          glAtomID[iTo]    = refMagnet.glAtomID[iTo];
          glSpins[iTo]       = new double [DIMENSIONS];
          for (int iDi=0;iDi<DIMENSIONS;iDi++)
              glSpins[iTo][iDi] = refMagnet.glSpins[iTo][iDi];
    }    }
}

//**********************************************************************
// Copy constructor 2: Copy Magnet data and replicate unit cell nXnXn.
Magnet::Magnet(int n, const Magnet &refMagnet) : Crystal(n,refMagnet)
{
 nTot      = 0;
 glSpeciesID = NULL;
 glAtomID    = NULL;
 glSpins     = NULL;

 // There's no point in replicating a default (empty) refMagnet 
 // - just return the default constructor.
 if (refMagnet.xtlData.getNumSpecies()==0) return;

 // replicate Crystal data
 xtlData=Crystal(n,refMagnet.xtlData);

 // Explicitly derive global atomic data from xtlData.
 reformatCrystal();

 // replicate spins
 int n0 = refMagnet.nTot;
 for (int iBox=1;iBox<n*n*n;iBox++)
 for (int jTo=0;jTo<nTot;jTo++)
 for (int kg=0;kg<DIMENSIONS;kg++)
     glSpins[iBox*n0+jTo][kg] = refMagnet.glSpins[jTo][kg];

}

//**********************************************************************
Magnet::~Magnet()
{
// use default deconstructor for xtlData.

// Delete global data
if (nTot>0)
   {
    delete [] glSpeciesID;
    delete [] glAtomID;
    for (int iTo=0;iTo<nTot;iTo++) delete [] glSpins[iTo];
    delete [] glSpins;
   }
}

//**********************************************************************
const Magnet &Magnet::operator =(const Magnet &rhs)
{
 xtlData=rhs.xtlData;
 glSpeciesID = NULL;
 glAtomID    = NULL;
 glSpins     = NULL;

 nTot = rhs.nTot;

cout << "       nTot " << nTot << endl;

 if (nTot>0)
    {
     glSpeciesID = new int [nTot];
     glAtomID    = new int [nTot];
     glSpins     = new double* [nTot];
     for (int iTo=0;iTo<nTot;iTo++)
         {
          glSpeciesID[iTo] = rhs.glSpeciesID[iTo];
          glAtomID[iTo]    = rhs.glAtomID[iTo];
          glSpins[iTo]       = new double [DIMENSIONS];
          for (int iDi=0;iDi<DIMENSIONS;iDi++)
              glSpins[iTo][iDi] = rhs.glSpins[iTo][iDi];
    }    }

 return *this;
}

//**********************************************************************
// overload equality operator
bool Magnet::operator ==(const Magnet &rhs)
{
 if (xtlData!=rhs.xtlData) return false;

 if (nTot!=rhs.nTot) return false;

 const double tol = 1.e-6;
 for (int iTo=0;iTo<nTot;iTo++)
     {
      if (glSpeciesID[iTo]!=rhs.glSpeciesID[iTo]) return false;
      if (glAtomID[iTo]   !=rhs.glAtomID[iTo])    return false;
      for (int iDi=0;iDi<DIMENSIONS;iDi++)
          if(fabs(glSpins[iTo][iDi]-rhs.glSpins[iTo][iDi])>tol)
            return false;
     }
 return true;
}

//**********************************************************************
// overload inequality operator
bool Magnet::operator !=(const Magnet &rhs)
{
 return !(*this==rhs);
}

//**********************************************************************
// copy/rearrange atomic data from Crystal format to global Magnet
int Magnet::reformatCrystal()
{
 if (xtlData.getNumSpecies()==0) return 0;

 assert(glSpeciesID==NULL && glAtomID==NULL && glSpins==NULL);

 nTot = 0;
 for (int iSp=0;iSp<xtlData.getNumSpecies();iSp++)
     {
      nTot += xtlData.getNumAtoms(iSp);
     }

 glSpeciesID = new int [nTot];
 glAtomID    = new int [nTot];
 glSpins     = new double* [nTot];

cout << " refoCHJCHHCCH " << endl;

 int iCount = 0;
 for (int iSp=0;iSp<xtlData.getNumSpecies();iSp++)
 for (int jAt=0;jAt<xtlData.getNumAtoms(iSp);jAt++)
     {
//cout << iSp << " " << jAt << " " << iCount << " " << nTot << endl;
      assert (iCount<nTot);
      glSpeciesID[iCount] = iSp;
      glAtomID[iCount]    = jAt;
      glSpins[iCount]     = new double[DIMENSIONS];
      iCount++;
     }

cout << glSpeciesID[0] << " " << glAtomID[0] << " " << endl;

cout << " refoDIDIDIDID \n";
 return 1;
}

//**********************************************************************
// Make list of nbr shells around each atom
int Magnet::makeNbrLists(double rmax)
{
 double lattice[3][3];
 int nRhomb[3];

 findnRhomb(rmax,lattice,nRhomb);

 // stuff goes here

 return 0;
}
