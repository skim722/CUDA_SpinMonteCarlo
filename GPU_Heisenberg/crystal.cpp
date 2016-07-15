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
#include <assert.h>

using namespace std;

//*****************************************************************************
//Default constructor: JBS
Crystal::Crystal()
{
  nSpecies = 0;
  nameSpecies = NULL;
  nAtoms = NULL;
  basis = NULL;

  for (int ig=0;ig<LATTICED;ig++)
      for (int jg=0;jg<LATTICED;jg++)
		  lattice[ig][jg]=0.;

  for (int ig=0;ig<LATTICED;ig++)
      for (int jg=0;jg<LATTICED;jg++)
		  primLattice[ig][jg]=0.;
}

//*****************************************************************************
//Alternate constructor USING INPUT FILE: MPS
Crystal::Crystal(string filename)
{
  ifstream fin;
  fin.open(filename.c_str());
  if (fin.fail()) {cout << " bad crystal input file\n";exit(9);}

  int ig, jg, n;
  double vec1[LATTICED], scalar_in_Ang;
  double au_by_Ang = 0.5291772086;

  cout << endl << endl << filename << ":" <<  endl << endl;
  
  //LATTICE
  fin >> scalar_in_Ang;
  cout << "    scalar_in_Ang " << scalar_in_Ang << endl;
  for (ig=0;ig<LATTICED;ig++)
      {
       for (jg =0;jg<LATTICED;jg++)
           {fin>>vec1[jg]; vec1[jg]=vec1[jg]*scalar_in_Ang/au_by_Ang;}
       for (jg =0;jg<LATTICED;jg++)
           {
            lattice[ig][jg] = vec1[jg];
            primLattice[ig][jg] = vec1[jg];
           }
       cout << "   " << lattice[ig][0] << " " << lattice[ig][1] << " " << lattice[ig][2] << endl;
      }
  cout << endl;

  fin >> nSpecies;
  cout << "   nSpecies " << nSpecies << endl;

  nAtoms = new int [nSpecies];
  nameSpecies = new string [nSpecies];
  basis = new double ** [nSpecies];

  for (int iSp=0;iSp<nSpecies;iSp++)
      {
       fin >> nAtoms[iSp];
       fin >> nameSpecies[iSp];
       cout << "   nAtoms[] " << nAtoms[iSp] << " " << nameSpecies[iSp] << endl;
       basis[iSp] = new double* [nAtoms[iSp]];
       for (int jAt=0;jAt<nAtoms[iSp];jAt++)
           {
            basis[iSp][jAt] = new double [LATTICED];
            for (int kg=0;kg<LATTICED;kg++)
                {fin >> vec1[kg]; vec1[kg]=vec1[kg]*scalar_in_Ang/au_by_Ang;}
            cout << "   ";
            for (int kg=0;kg<LATTICED;kg++)
                {
                 basis[iSp][jAt][kg] = vec1[kg];
                 cout << basis[iSp][jAt][kg] << " ";
                }
                cout << endl;
           }
      }

  fin.close();

  cout << endl << endl;

}

//*****************************************************************************
//Copy constructor -- for instantiating objects with other objects: JBS
Crystal::Crystal(const Crystal &refCrystal)
{
  cout << "FVSF\n";
  nSpecies = refCrystal.nSpecies;

  if (nSpecies == 0) return;

  nAtoms = new int [nSpecies];
  for (int iSp=0;iSp<nSpecies;iSp++)
       nAtoms[iSp] = refCrystal.nAtoms[iSp];

  nameSpecies = new string [nSpecies];
  for (int iSp=0;iSp<nSpecies;iSp++)
       nameSpecies[iSp] = refCrystal.nameSpecies[iSp];

  for (int ig=0;ig<LATTICED;ig++)
      for (int jg =0;jg<LATTICED;jg++)
          {
           lattice[ig][jg]=refCrystal.lattice[ig][jg];
           primLattice[ig][jg]=refCrystal.primLattice[ig][jg];
          }

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
cout << "SKJGFHF\n";
}

//*****************************************************************************
//Copy constructor2 -- for instantiating supercell from other objects: MPS
Crystal::Crystal(int n1, int n2, int n3, const Crystal &refCrystal)
{
cout << " OQIREJFIO\n";
  // Copy and replicate a cell nXnXn times.
  int i0;
  double vecg[LATTICED];

  // Convert the lattice vectors to a supercell
      for (int jg =0;jg<LATTICED;jg++)
          {
           lattice[0][jg]=(double)n1*refCrystal.lattice[0][jg];
           lattice[1][jg]=(double)n2*refCrystal.lattice[1][jg];
           lattice[2][jg]=(double)n3*refCrystal.lattice[2][jg];
          }
      for (int jg =0;jg<LATTICED;jg++)
          {
           primLattice[0][jg]=refCrystal.primLattice[0][jg];
           primLattice[1][jg]=refCrystal.primLattice[1][jg];
           primLattice[2][jg]=refCrystal.primLattice[2][jg];
          }

  nSpecies = refCrystal.nSpecies;

  if (nSpecies == 0) return;

  nAtoms = new int [nSpecies];
  nameSpecies = new string [nSpecies];
  basis = new double** [nSpecies];

  for (int iSp=0;iSp<nSpecies;iSp++)
      {
       nameSpecies[iSp] = refCrystal.nameSpecies[iSp];
       nAtoms[iSp] = n1*n2*n3*refCrystal.nAtoms[iSp];

       basis[iSp] = new double* [nAtoms[iSp]];
       for (int jAt=0;jAt<nAtoms[iSp];jAt++)
           basis[iSp][jAt] = new double [LATTICED];
      }

cout << "KJSFJKFS\n";

  // Copy the atomic basis to the supercell
  int iCount[nSpecies];
  for (int i=0;i<nSpecies;i++)
      {
       iCount[i] = 0.;
      }

  for (int i=0;i<n1;i++)
      {
       for (int j=0;j<n2;j++)
           {
            for (int k=0;k<n3;k++)
                {
                 for (int m=0;m<LATTICED;m++)
                     vecg[m]=i*refCrystal.lattice[0][m]
  	                    +j*refCrystal.lattice[1][m]
                            +k*refCrystal.lattice[2][m];

                 for (int iSp=0;iSp<nSpecies;iSp++)
                     {
                      for(int ii=0; ii<refCrystal.nAtoms[iSp]; ii++)
                         {
                          for(int jj=0; jj<LATTICED; jj++)
                             basis[iSp][iCount[iSp]][jj] =
                                         refCrystal.basis[iSp][ii][jj]
                                        +vecg[jj];
                          assert(iCount[iSp]<nAtoms[iSp]);
                          iCount[iSp]++;
                         }
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
      {
       lattice[ig][jg]=rhs.lattice[ig][jg];
       primLattice[ig][jg]=rhs.primLattice[ig][jg];
      }

  nSpecies = rhs.nSpecies;

  if (nSpecies == 0) return *this;

  nameSpecies = new string [nSpecies];

  nAtoms = new int [nSpecies];
  for (int iSp=0;iSp<nSpecies;iSp++)
      {
       nameSpecies[iSp] = rhs.nameSpecies[iSp];
       nAtoms[iSp] = rhs.nAtoms[iSp];
      }

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
cout << " (#)UFKJKkjherkj\n";
  for (int ig=0;ig<LATTICED;ig++)
  for (int jg=0;jg<LATTICED;jg++)
      {
       if (lattice[ig][jg] != rhs.lattice[ig][jg]) return false;
       if (primLattice[ig][jg] != rhs.primLattice[ig][jg]) return false;
      }

cout << " (2)UFKJKkjherkj\n";
  if(nSpecies != rhs.nSpecies) return false;

  for(int iSp=0; iSp<nSpecies; iSp++)
     {
      if (nameSpecies[iSp] != rhs.nameSpecies[iSp]) return false;
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
cout << " (#)UFKJKkjherkj\n";
  return true;
}

//*****************************************************************************
//Overload inequality operator
bool Crystal::operator !=(const Crystal &rhs)
{
  return !(*this == rhs);
}

//*****************************************************************************
//Deallocate Memory
int Crystal::deallocateMemory()
{
  //verify memory has been allocated
  if (nameSpecies != NULL) delete [] nameSpecies;

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

