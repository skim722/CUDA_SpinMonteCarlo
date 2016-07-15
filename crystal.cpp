/*******************************************************/
/* crystal.cpp -- Class definitions for crystal.       */
/*                (c) J.B. Sturgeon, October, 24, 2006 */
/* modified 6/27/14 MPS to allow multiple species.     */
/*******************************************************/
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include "crystal.h"
#include <cmath>

using namespace std;

//*****************************************************************************
//Default constructor: JBS
Crystal::Crystal()
{
  nSpecies = 0;
  nAtoms = NULL;
  basis = NULL;
  for (int ig=0;ig<LATTICED;ig++)
      for (int jg=0;jg<LATTICED;jg++)
		  lattice[ig][jg]=0.;
}

//*****************************************************************************
//Alternate constructor USING INPUT FILE: MPS
Crystal::Crystal(string filename)
{
  ifstream fin;
  fin.open(filename.c_str());

  int ig, jg, n;
  double vec1[LATTICED], scalar;
  
//LATTICE
  fin >> scalar;
  cout << "scalar " << scalar << endl;
  for (ig=0;ig<LATTICED;ig++)
      {
       for (jg =0;jg<LATTICED;jg++)
		   {fin>>vec1[jg]; vec1[jg]=vec1[jg]*scalar;}
       setIthLattice(ig,vec1);
      }

  fin >> nSpecies;
  cout << "nSpecies " << nSpecies << endl;

  nAtoms = new int [nSpecies];

//BASIS  
  basis = new double ** [nSpecies];

  for (int iSp=0;iSp<nSpecies;iSp++)
      {
       fin >> nAtoms[iSp];
       cout << "nAtoms[] " << nAtoms[iSp] << endl;
       basis[iSp] = new double* [nAtoms[iSp]];
       for (int jAt=0;jAt<nAtoms[iSp];jAt++)
           {
            basis[iSp][jAt] = new double [LATTICED];
            for (int kg=0;kg<LATTICED;kg++)
            {fin >> vec1[kg]; vec1[kg]=vec1[kg]*scalar;}
            setIJthBasis(iSp,jAt,vec1);
           }
      }

  fin.close();

}

//*****************************************************************************
//Copy constructor -- for instantiating objects with other objects: JBS
Crystal::Crystal(const Crystal &refCrystal)
{
  nSpecies = refCrystal.nSpecies;

  if (nSpecies == 0) return;

  nAtoms = new int [nSpecies];
  for (int iSp=0;iSp<nSpecies;iSp++)
       nAtoms[iSp] = refCrystal.nAtoms[iSp];

  for (int ig=0;ig<LATTICED;ig++)
      for (int jg =0;jg<LATTICED;jg++) lattice[ig][jg]=refCrystal.lattice[ig][jg];

  basis = new double ** [nSpecies];
  for (int iSp=0;iSp<nSpecies;iSp++)
      {
       basis[iSp] = new double * [nAtoms[iSp]];
       for(int ii=0; ii<nAtoms[iSp]; ii++)
          {
           basis[iSp][ii] = new double [LATTICED];
           for(int jj=0; jj<LATTICED; jj++) basis[iSp][ii][jj] = refCrystal.basis[iSp][ii][jj];
          }
      }
}

//*****************************************************************************
//Copy constructor2 -- for instantiating supercell from other objects: MPS
Crystal::Crystal(int n, const Crystal &refCrystal)
{
  // Copy and replicate a cell nXnXn times.
  int i0;
  double vecg[LATTICED];

  // Convert the lattice vectors to a supercell
  for (int ig=0;ig<LATTICED;ig++)
      for (int jg =0;jg<LATTICED;jg++)
          lattice[ig][jg]=(double)n*refCrystal.lattice[ig][jg];

  nSpecies = refCrystal.nSpecies;

  if (nSpecies == 0) return;

  nAtoms = new int [nSpecies];
  basis = new double** [nSpecies];

  for (int iSp=0;iSp<nSpecies;iSp++)
      {
       nAtoms[iSp] = n*n*n*refCrystal.nAtoms[iSp];

       basis[iSp] = new double* [nAtoms[iSp]];
      }

  // Copy the atomic basis to the supercell
  int iCount = 0;
  for (int i=0;i<n;i++)
      {
       for (int j=0;j<n;j++)
           {
            for (int k=0;k<n;k++)
                {
                 for (int m=0;m<LATTICED;m++)
                     vecg[m]=i*refCrystal.lattice[0][m]
  	                    +j*refCrystal.lattice[1][m]
                            +k*refCrystal.lattice[2][m];

                 for (int iSp=0;iSp<nSpecies;iSp++)
                     {
                      for(int ii=0; ii<refCrystal.nAtoms[iSp]; ii++)
                      for(int jj=0; jj<LATTICED; jj++)
                      basis[iSp][iCount][jj] = refCrystal.basis[iSp][ii][jj]
                                           +vecg[jj];

                      iCount+=refCrystal.nAtoms[iSp];
                     }
                }
            }
       }
}

//*****************************************************************************
//Default destructor
Crystal::~Crystal()
{
  deallocateMemory();
}

//*****************************************************************************
//Overload assignment operator
//  crystal1 = crystal2;
const Crystal &Crystal::operator =(const Crystal &rhs)
{
  if(&rhs != this) { //check for self-assignment and avoid

  if (nSpecies !=0)
      deallocateMemory();

  for (int ig=0;ig<LATTICED;ig++)
  for (int jg =0;jg<LATTICED;jg++)
      lattice[ig][jg]=rhs.lattice[ig][jg];

  nSpecies = rhs.nSpecies;

  if (nSpecies == 0) return *this;

  nAtoms = new int [nSpecies];
  for (int iSp=0;iSp<nSpecies;iSp++)
       setNumAtoms(iSp,rhs.nAtoms[iSp]);

  basis = new double ** [nSpecies];
    for(int iSp=0; iSp<nSpecies; iSp++)
       {
        basis[iSp] = new double * [nAtoms[iSp]];
        for (int jAt=0;jAt<nAtoms[iSp];jAt++)
            {
             basis[iSp][jAt] = new double [LATTICED];
             for (int kg=0;kg<LATTICED;kg++)
                  basis[iSp][jAt][kg] = rhs.basis[iSp][jAt][kg];
            }
       }
  }

  return *this;  //enables x = y = z;
}

//*****************************************************************************
//Overload equality operator
bool Crystal::operator ==(const Crystal &rhs)
{
  for (int ig=0;ig<LATTICED;ig++)
  for (int jg=0;jg<LATTICED;jg++)
      if (lattice[ig][jg] != rhs.lattice[ig][jg]) return false;

  if(nSpecies != rhs.nSpecies) return false;

  for(int iSp=0; iSp<nSpecies; iSp++)
     {
      if (nAtoms[iSp] != rhs.nAtoms[iSp]) return false;
      for(int ii=0; ii<nAtoms[iSp]; ii++)
         {
          for(int jj=0; jj<LATTICED; jj++)
             {
              if(basis[iSp][ii][jj] != rhs.basis[iSp][ii][jj])
                 return false;
             }
         }
     }
  return true;
}

//*****************************************************************************
//Overload inequality operator
bool Crystal::operator !=(const Crystal &rhs)
{
  return !(*this == rhs);
}

//*****************************************************************************
//Allocate memory
int Crystal::allocateMemory()
{
  try
  {
    if(nSpecies > 0 && nAtoms != NULL)
      {
       if (basis != NULL) { cout << "basis not null!\n";exit(14);}
       basis = new double** [nSpecies];

       for (int iSp=0;iSp<nSpecies;iSp++)
           {
            basis[iSp] = new double* [nAtoms[iSp]];
            for (int jAt=0;jAt<nAtoms[iSp];jAt++)
                 basis[iSp][jAt] = new double [LATTICED];
           }

       if(basis == NULL) {
         cout << "ERROR(crystal.cpp): Not enough memory for basis.\n" << endl;
         throw;}
       }
	    else {cout << "WARNING(crystal.cpp): nAtoms =0, arrays are NULL.\n" << endl;}
  }

  catch ( ... )  //Catch all exceptions
  {
    if( basis == NULL ) {
      cout << "ERROR(crystal.cpp): Unable to allocate memory for basis array.\n" << endl;
    }
    else {
      //Other error messages here?
      ;
    }

    //Possibly write code to end program here, e.g.
    //exitMyProgram();

    return -1;
  }

  return 0;
}

//*****************************************************************************
//Deallocate Memory
int Crystal::deallocateMemory()
{
  //verify memory has been allocated

  if(nSpecies!=0 && nAtoms!=NULL) {
    for (int iSp=0;iSp<nSpecies;iSp++)
        {
         for (int jAt=0;jAt<nAtoms[iSp];jAt++)
              delete [] basis[iSp][jAt];
         delete [] basis[iSp];
        }
    delete [] basis;
    delete [] nAtoms;
  }
  basis = NULL;
  nAtoms = NULL;
  nSpecies = 0;

  return 0;
}


//*****************************************************************************
  int Crystal::setNumAtoms(int iSp, int n) {nAtoms[iSp]=n;}
//*****************************************************************************
int Crystal::setBasisToZero()
{
  if(nSpecies>0  && nAtoms !=NULL &&  basis!=NULL) {
    for(int iSp=0; iSp<nSpecies; iSp++)
      for(int jAt=0;jAt<nAtoms[iSp];jAt++)
        for(int kk=0; kk<LATTICED; kk++)
          {basis[iSp][jAt][kk] = 0.0;}
  }

  return 0;
}
