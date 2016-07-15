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
#include "KKR.h"

using namespace std;

double mymax(double a, double b) {return (a>b) ?a:b;}

//*****************************************************************************
int translate(double primLatt[][LATTICED],double primInv[][LATTICED], 
              double vecg[LATTICED], double vec0[LATTICED])
{
 // Translate atom at vecg back to the unit cell centered at 000.
 // Use this to identify an atom anywhere in the lattice with
 // an atomic site in the unit cell.
 // This mapping can be flaky at the boundary, here +/-0.5
 // If I translate two points separately and then subtract,
 // I don't necessarily get the smallest possible distance
 // between replicas of the two points.
 // I can fix this taking the difference between 2 points first,
 // then translating the difference into the unit cell centered at 000.
 // If the two points coincide on the lattice, 
 // then their difference will map to (000).

 double Xtl[LATTICED];

 // First, transform to lattice coordinates
 dot(primInv,vecg, Xtl);

 // Second, cancel out integer lattice displacements.
 Xtl[0] -= (double)rint(Xtl[0]+0.e-8);
 Xtl[1] -= (double)rint(Xtl[1]+0.e-8);
 Xtl[2] -= (double)rint(Xtl[2]+0.e-8);

 // Third, convert back to Cartesian coordinates
 dot(primLatt,Xtl,vec0);
}

// Assume there are fewer than maxNbr shells and
// fewer than maxNbr neighbors in each shell.
const static int maxNbr = 2000;

int makeListShellRad2(double dist2, double rmax2, int& nSh, int maxSh,
                  double rSh2[], double tol);
int myInsert2(int iTo, int jTo,
              string glNameI, double glCoori[3], 
              string glNameJ, double glCoorj[3], double delrCryst[3],
              int i, int j, int k, double dist2, double rmax2,
              int nSh, double shellRadius2[],
              int nNbr[], int maxnbr,
              int** iNbr, double** JNbr, 
              double primLattice[][LATTICED], double primInv[][LATTICED], int nLines,
              string name1[], double tau1[][3], string name2[], double tau2[][3], 
              double del[][3], double rsh[], int nnn[][3], double delNnn[][3], double JmeV[],
              double tol, int&iLi);

int findnRhomb(double rmax, double lattice[3][3], int nRhomb[3]);

//**********************************************************************
// Default constructor inherits default Crystal, leaves arrays undefined
Magnet::Magnet() : Crystal()
{
 nTot      = 0;
 glSpeciesID = NULL;
 glAtomID    = NULL;
 glName      = NULL;
 glCoord     = NULL;
 glSpins     = NULL;

 nShells     = NULL;

 xtlData=Crystal();
 kkr    = KKR();

 int mmm[3];
 kkr.getNnn(0,mmm);

 setRmax();
cout << " hoo1 " << mmm[0] << " " << mmm[1] << " " << mmm[2] << endl;

T = 0.;
H[0] = H[1] = H[2] = 0.;
nMonte=0;

}

//**********************************************************************
// Constructor from multiple input files, including crystal data.
Magnet::Magnet(string cName, string mName) : Crystal()
{
 nTot      = 0;
 glSpeciesID = NULL;
 glName      = NULL;
 glAtomID    = NULL;
 glCoord     = NULL;
 glSpins     = NULL;

 nShells     = NULL;

cout << "FSFS\n";
xtlData=Crystal(cName);

 kkr    = KKR();

 int mmm[3];
 mmm[0] = 0;
 mmm[1] = 0;
 mmm[2] = 0;
 kkr.getNnn(0,mmm);

 cout << " got KKR " << mmm[0] << " " << mmm[1] << " " << mmm[2] << endl << endl;

 // convert unit cell xtlData to "global" format.
 reformatCrystal();

 ifstream fin;
 fin.open(mName.c_str());

 double dum;
 for (int iTo=0;iTo<nTot;iTo++)
     {fin >> glSpins[iTo][0];
      fin >> glSpins[iTo][1];
      fin >> glSpins[iTo][2];}

 fin.close();

 double rmax = kkr.getRmax();
 // Set rmax based on kkr cutoff and
 // verify that kkr data agrees with Crystal.
 int nLines = kkr.getNLines();
     secondPass(nTot,nLines,glSpeciesID,glAtomID,
                  nShells,shellRadius2,
                  shellSize,shellNbr,rmax);

 // construct neighbor lists for summing J terms

 rmax = kkr.getRmax();

 kkr.getNnn(0,mmm);

 for (int iSp=0;iSp<getNumSpecies();iSp++) cout <<getNumAtoms(iSp)<< " " ;

cout << " SJKF\n";
 makeNbrLists(kkr.getRmax());

T = 0.;
H[0] = H[1] = H[2] = 0.;

}

//**********************************************************************
// Copy constructor.
Magnet::Magnet(const Magnet &refMagnet) : Crystal()
{
 nTot      = 0;
 glSpeciesID = NULL;
 glName      = NULL;
 glAtomID    = NULL;
 glCoord     = NULL;
 glSpins     = NULL;
 nShells     = NULL;
 // copy Crystal data
 cout << "SFVSdd\n";
 xtlData=refMagnet.xtlData;

 // directly copy all global data
 nTot = refMagnet.nTot;

 cout << "SFVSdd\n";

 if (nTot==0) return;

     glSpeciesID = new int [nTot];
     glName      = new string [nTot];
     glAtomID    = new int [nTot];
     glCoord     = new double* [nTot];
     glSpins     = new double* [nTot];
     for (int iTo=0;iTo<nTot;iTo++)
         {
          glSpeciesID[iTo] = refMagnet.glSpeciesID[iTo];
          glName[iTo]      = refMagnet.glName[iTo];
          glAtomID[iTo]    = refMagnet.glAtomID[iTo];
          glCoord[iTo]       = new double [LATTICED];
          glSpins[iTo]       = new double [DIMENSIONS];
          for (int iDi=0;iDi<DIMENSIONS;iDi++)
              glSpins[iTo][iDi] = refMagnet.glSpins[iTo][iDi];
          for (int iLa=0;iLa<LATTICED;iLa++)
              glCoord[iTo][iLa] = refMagnet.glCoord[iTo][iLa];
    }

 cout << "SFVSdd\n";

 kkr = refMagnet.kkr;

 cout << "SFVSdd\n";

 cout << " This is not the most efficient copy 1 !\n";

 rmax = refMagnet.rmax;

//cout << " IOEREGIO\n";
// makeNbrLists(rmax);
//cout << " IOEREGIO\n";

 nShells      = new int [nTot];
 shellSize    = new int* [nTot];
 shellRadius2 = new double* [nTot];
 shellNbr     = new int**[nTot];
 J            = new double** [nTot];


 for (int iTo=0;iTo<nTot;iTo++)
     {
      nShells[iTo] = refMagnet.nShells[iTo];

cout << nShells[iTo] << "  ";

      shellSize[iTo]    = new int     [nShells[iTo]];
      shellRadius2[iTo] = new double  [nShells[iTo]];
      shellNbr[iTo]     = new int*    [nShells[iTo]];
      J       [iTo]     = new double* [nShells[iTo]];

      for (int jSh=0;jSh<nShells[iTo];jSh++)
          {
           shellSize[iTo][jSh] = refMagnet.shellSize[iTo][jSh];
           shellRadius2[iTo][jSh] = refMagnet.shellRadius2[iTo][jSh];
//cout << shellSize[iTo][jSh] << " " << shellRadius2[iTo][jSh] << " ";
//cout << refMagnet.shellSize[iTo][jSh] << " " << refMagnet.shellRadius2[iTo][jSh] << endl;
           shellNbr[iTo][jSh]     = new int    [shellSize[iTo][jSh]];
           J       [iTo][jSh]     = new double [shellSize[iTo][jSh]];

           for (int kAt=0;kAt<shellSize[iTo][jSh];kAt++)
               {
                shellNbr[iTo][jSh][kAt] = refMagnet.shellNbr[iTo][jSh][kAt];
                J[iTo][jSh][kAt] = refMagnet.J[iTo][jSh][kAt];
//cout << J[iTo][jSh][kAt] << endl;
               }
          }
     }


T =refMagnet.getT(); 
refMagnet.getH(H);
nMonte = refMagnet.getnMonte();
 
}

//**********************************************************************
// Copy constructor 2: Copy Magnet data and replicate unit cell nXnXn.
Magnet::Magnet(int n1, int n2, int n3, const Magnet &refMagnet) : Crystal()
{
 nTot      = 0;
 glSpeciesID = NULL;
 glName      = NULL;
 glAtomID    = NULL;
 glCoord     = NULL;
 glSpins     = NULL;
 nShells     = NULL;

 // There's no point in replicating a default (empty) refMagnet 
 // - just return the default constructor.
 if (refMagnet.xtlData.getNumSpecies()==0) return;

 // replicate Crystal data
 cout << "JKEFJEKF\n";
 xtlData = Crystal(n1,n2,n3,refMagnet.xtlData);
 double vecg[3];
 for (int i=0;i<xtlData.getNumAtoms(0);i++)
     {
      xtlData.getIJthBasis(0,i,vecg);
     }

cout << "JKEFJEK2\n";

 // Explicitly derive global atomic data from xtlData.
 reformatCrystal();

cout << "JKEFJEK3\n";

 // copy kkr data
 kkr = refMagnet.kkr;

cout << "JKEFJEK4\n";

 rmax = kkr.getRmax();

 // replicate spins
 int n0 = refMagnet.nTot;
 glSpins = new double* [n0*n1*n2*n3];
 for (int i=0;i<n0*n1*n2*n3;i++) glSpins[i] = new double [DIMENSIONS];
 for (int iBox=0;iBox<n1*n2*n3;iBox++)
 for (int jTo=0;jTo<n0;jTo++)
     {
      for (int kg=0;kg<DIMENSIONS;kg++)
          {
           glSpins[iBox*n0+jTo][kg] = refMagnet.glSpins[jTo][kg];
          }
      }
cout << "JKEFJEK5\n";

 rmax = kkr.getRmax();

 int nRhomb[3];
 double lattice[3][3];
 xtlData.getIthLattice(0,lattice[0]);
 xtlData.getIthLattice(1,lattice[1]);
 xtlData.getIthLattice(2,lattice[2]);

 findnRhomb(rmax,lattice,nRhomb);

 if (2*n1+1<nRhomb[0] || 2*n2+1<nRhomb[1] || 2*n3+1<nRhomb[2])
    {
     cout << "replicate a large enough supercell to encompass rmax "
          << rmax << endl;
     exit(14);
    }

cout << rmax << " fdjkfjdkjkdf\n";
 makeNbrLists(rmax);
cout << " fdjkfjdkjkdf\n";

T =refMagnet.getT(); 
refMagnet.getH(H);
nMonte = 0;

}

//**********************************************************************
Magnet::~Magnet()
{
 // use default destructor for xtlData.
cout << "QUREFHIUHF\n";

 // Delete global data
 if (nTot>0)
    {
     delete [] glSpeciesID;
     delete [] glAtomID;
     delete [] glName;
     for (int iTo=0;iTo<nTot;iTo++)
         {
          delete [] glCoord[iTo];
          delete [] glSpins[iTo];
         }
     delete [] glCoord;
     delete [] glSpins;
    }

cout << "KLSDKLDSKLDSLKDSKLDS\n";

 if (nTot>0 && nShells!=NULL)
    {
     for (int iTo=0;iTo<nTot;iTo++)
         {
          for (int jSh=0;jSh<nShells[iTo];jSh++)
              {
               delete [] shellNbr[iTo][jSh];
               delete [] J[iTo][jSh];
              }
          delete [] shellSize[iTo];
          delete [] shellRadius2[iTo];
          delete [] shellNbr[iTo];
          delete [] J[iTo];
         }
     delete [] nShells;
     delete [] shellRadius2;
     delete [] shellSize;
     delete [] shellNbr;
     delete [] J;
    }

}

//**********************************************************************
const Magnet &Magnet::operator =(const Magnet &rhs)
{
 assert(0==9); // not vetted
 xtlData=rhs.xtlData;
 glSpeciesID = NULL;
 glAtomID    = NULL;
 glCoord     = NULL;
 glSpins     = NULL;
 nShells     = NULL;

 nTot = rhs.nTot;

 if (nTot>0)
    {
     glSpeciesID = new int [nTot];
     glAtomID    = new int [nTot];
     glName      = new string  [nTot];
     glCoord     = new double* [nTot];
     glSpins     = new double* [nTot];

     for (int iTo=0;iTo<nTot;iTo++)
         {
          glSpeciesID[iTo] = rhs.glSpeciesID[iTo];
          glAtomID[iTo]    = rhs.glAtomID[iTo];
          glName[iTo]      = rhs.glName[iTo];
          glCoord[iTo]       = new double [LATTICED];
          glSpins[iTo]       = new double [DIMENSIONS];
          for (int iDi=0;iDi<DIMENSIONS;iDi++)
              glSpins[iTo][iDi] = rhs.glSpins[iTo][iDi];
          for (int iLa=0;iLa<LATTICED;iLa++)
              glCoord[iTo][iLa] = rhs.glCoord[iTo][iLa];
    }    }

cout << kkr.getNLines() << endl;

 kkr = rhs.kkr;

 nShells = new int [nTot];
 shellRadius2 = new double* [nTot];
 shellSize    = new int* [nTot];
 shellNbr     = new int**[nTot];
 J            = new double** [nTot];


 for (int iTo=0;iTo<nTot;iTo++)
     {
      nShells[iTo] = rhs.nShells[iTo];

      shellSize[iTo]    = new int     [nShells[iTo]];
      shellRadius2[iTo] = new double  [nShells[iTo]];
      shellNbr[iTo]     = new int*    [nShells[iTo]];
      J       [iTo]     = new double* [nShells[iTo]];

      for (int jSh=0;jSh<nShells[iTo];jSh++)
          {
           shellSize[iTo][jSh] = rhs.shellSize[iTo][jSh];
           shellRadius2[iTo][jSh] = rhs.shellRadius2[iTo][jSh];
           shellNbr[iTo][jSh]     = new int    [shellSize[iTo][jSh]];
           J       [iTo][jSh]     = new double [shellSize[iTo][jSh]];

           for (int kAt=0;kAt<shellSize[iTo][jSh];kAt++)
               {
                shellNbr[iTo][jSh][kAt] = rhs.shellNbr[iTo][jSh][kAt];
                J[iTo][jSh][kAt] = rhs.J[iTo][jSh][kAt];
               }
          }
     }

// cout << " This is not the most efficient copy!\n";
// makeNbrLists(kkr.getRmax());
cout << " OK5\n";

/*
 for (int iTo=0;iTo<nTot;iTo++)
     {
      nShells[iTo]=rhs.nShells[iTo];
      for (int jSh=0;jSh<nShells[iTo];iTo++)
          {
           shellSize[iTo][jSh]=rhs.shellSize[iTo][jSh];
           shellRadius2[iTo][jSh]=rhs.shellRadius2[iTo][jSh];
           for (int kAt=0;kAt<shellSize[iTo][jSh];kAt++)
               {
                shellNbr[iTo][jSh][kAt]=rhs.shellNbr[iTo][jSh][kAt];
                J[iTo][jSh][kAt]=rhs.J[iTo][jSh][kAt];
               }
          }
     }
*/

T= rhs.getT();
rhs.getH(H);
nMonte=rhs.getnMonte();

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
      if (glName[iTo]     !=rhs.glName[iTo])      return false;
      for (int iDi=0;iDi<DIMENSIONS;iDi++)
          if(fabs(glSpins[iTo][iDi]-rhs.glSpins[iTo][iDi])>tol)
      for (int iLa=0;iLa<LATTICED;iLa++)
          if(fabs(glCoord[iTo][iLa]-rhs.glSpins[iTo][iLa])>tol)
            return false;
     }

 if (kkr!=rhs.kkr) return false;

 for (int iTo=0;iTo<nTot;iTo++)
     {
      if (nShells[iTo]!=rhs.nShells[iTo]) return false;
      for (int jSh=0;jSh<nShells[iTo];jSh++)
          {
           if (shellSize[iTo][jSh]!=rhs.shellSize[iTo][jSh]) return false;
           if (shellRadius2[iTo][jSh]!=rhs.shellRadius2[iTo][jSh]) return false;
           for (int kAt=0;kAt<shellSize[iTo][jSh];kAt++)
               {
                if (shellNbr[iTo][jSh][kAt]!=rhs.shellNbr[iTo][jSh][kAt]) return false;
                if (J[iTo][jSh][kAt]!=rhs.J[iTo][jSh][kAt]) return false;
               }
          }
     }

if (T!=rhs.getT()) return false;
double Hdum[3];
rhs.getH(Hdum);
if (Hdum[0]!=H[0] || Hdum[1]!=H[1] || Hdum[2]!=H[2]) return false;
if (nMonte!=rhs.getnMonte()) return false;


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

 assert(glSpeciesID==NULL && glName==NULL && glAtomID==NULL && glSpins==NULL);

 nTot = 0;
 for (int iSp=0;iSp<xtlData.getNumSpecies();iSp++)
     {
      nTot += xtlData.getNumAtoms(iSp);
     }

 glSpeciesID = new int [nTot];
 glName      = new string [nTot];
 glAtomID    = new int [nTot];
 glCoord     = new double* [nTot];
 glSpins     = new double* [nTot];

 int iCount = 0;
 for (int iSp=0;iSp<xtlData.getNumSpecies();iSp++)
 for (int jAt=0;jAt<xtlData.getNumAtoms(iSp);jAt++)
     {
      assert (iCount<nTot);
      glSpeciesID[iCount] = iSp;
      glName[iCount]      = xtlData.getSpeciesName(iSp);
      glAtomID[iCount]    = jAt;
      glSpins[iCount]     = new double[DIMENSIONS];
      glCoord[iCount]     = new double[LATTICED];
      double vecg[LATTICED];
      xtlData.getIJthBasis(iSp,jAt,vecg);
      for (int iLa=0;iLa<LATTICED;iLa++) glCoord[iCount][iLa] = vecg[iLa];
      iCount++;
     }

 return 1;
}

//**********************************************************************
// Make list of nbr shells around each atom
int Magnet::makeNbrLists(double  rmax)
{
 double lattice[3][3];
 xtlData.getIthLattice(0,lattice[0]);
 xtlData.getIthLattice(1,lattice[1]);
 xtlData.getIthLattice(2,lattice[2]);

 double primLattice[3][3];
 xtlData.getIthPrimLattice(0,primLattice[0]);
 xtlData.getIthPrimLattice(1,primLattice[1]);
 xtlData.getIthPrimLattice(2,primLattice[2]);

 // define inverse of primitive lattice
 // used to transform vectors from Cartesian to lattice coordinates.
 double a[LATTICED][LATTICED], primInv[LATTICED][LATTICED];
 inverse(primLattice,a);
 transpose(a,primInv);


 int nRhomb[3];
 findnRhomb(rmax,lattice,nRhomb);


 double rmax2 = rmax*rmax;

  cout << "rmax " << rmax << " nRhomb " << nRhomb[0] << " " << nRhomb[1] << " " << nRhomb[2] << endl << endl;
if (nRhomb[0]<0 || nRhomb[1] << 0 || nRhomb[2] << 0)
   {
    cout << lattice[0][0] << " " << lattice[0][1] << " " << lattice[0][2] << endl;
    cout << lattice[1][0] << " " << lattice[1][1] << " " << lattice[1][2] << endl;
    cout << lattice[2][0] << " " << lattice[2][1] << " " << lattice[2][2] << endl;

    cout << primLattice[0][0] << " " << primLattice[0][1] << " " << primLattice[0][2] << endl;
    cout << primLattice[1][0] << " " << primLattice[1][1] << " " << primLattice[1][2] << endl;
    cout << primLattice[2][0] << " " << primLattice[2][1] << " " << primLattice[2][2] << endl;
   }
//exit(9);

 nShells = new int [nTot];
 shellRadius2 = new double* [nTot];
 shellSize    = new int* [nTot];
 shellNbr     = new int**[nTot];
 J            = new double** [nTot];

 // Copy KKR data:
 int nLines = kkr.getNLines();
 int nnn[nLines][3];
 double JmeV[nLines];
 double tau1[nLines][3], tau2[nLines][3], rsh[nLines], del[nLines][3];
 string name1[nLines], name2[nLines];
 for (int iLi=0;iLi<nLines;iLi++)
     {
      kkr.getTau1(iLi,tau1[iLi]);
      kkr.getTau2(iLi,tau2[iLi]);
      kkr.getDel (iLi,del[iLi]);
      rsh[iLi] = kkr.getRsh(iLi);
      kkr.getNnn(iLi,nnn[iLi]);
      name1[iLi] = kkr.getName1(iLi);
      name2[iLi] = kkr.getName2(iLi);
      JmeV[iLi] = kkr.getJmeV(iLi);
     }

 // Assume there are fewer than maxShell shells.
 const int maxShell=10000;
 double rSh2[maxShell];

 // zeroth pass:
 // loop over sites, compute # of shells inside rmax
 for (int iTo=0;iTo<nTot;iTo++)
     {
      // First shell is always the origin.
      nShells[iTo] = 1;
      rSh2[0] = 0.;
     }

cout << nRhomb[0] << " " << nRhomb[1] << " " << nRhomb[2] << " AAAAA\n"<< endl;
//exit(9);

 // First pass:
 // loop over sites, compute # of shells inside rmax
 for (int iTo=0;iTo<nTot;iTo++)
     {
      // Loop over replica cells and atomic basis.
      // vecg is offset vector, delrCryst is vector between 2 atoms
      double wecg[LATTICED], xecg[LATTICED], delrCryst[LATTICED];
      // loop over cells guaranteed to encompass rmax
      for (int i=-nRhomb[0];i<=nRhomb[0];i++)
          {
           for (int j=-nRhomb[1];j<=nRhomb[1];j++)
               {
                vscopy(LATTICED,double(i),lattice[0],wecg);
                vaxpy(LATTICED,double(j),lattice[1],wecg);
                for (int k=-nRhomb[2];k<=nRhomb[2];k++)
                    {
                     vscopy(LATTICED,1.,wecg,xecg);
                     vaxpy(LATTICED,double(k),lattice[2],xecg);
                     for (int jTo=0;jTo<nTot;jTo++)
                         {
                          vscopy(LATTICED,-1.,glCoord[iTo],delrCryst);
                          vaxpy(LATTICED,+1.,glCoord[jTo],delrCryst);
                          vaxpy(LATTICED,+1.,xecg,delrCryst);
                          double dist2 = mydot(delrCryst,delrCryst);

                          makeListShellRad2(dist2,rmax2,nShells[iTo],maxShell,rSh2,1.e-6);
                         }
                    }
               }
          }

     // Now nShells[iTo] = # of nbr shells around iTo'th site
     shellRadius2[iTo] = new double [nShells[iTo]];
     for (int jSh=0;jSh<nShells[iTo];jSh++)
          shellRadius2[iTo][jSh] = rSh2[jSh];
     shellSize[iTo] = new int [nShells[iTo]];
     shellNbr[iTo]  = new int* [nShells[iTo]];
     J[iTo]         = new double* [nShells[iTo]];

     }
cout << " etoiwuhRRR\n";

 // pre compute conversion from lattice vectors nnn to Cartesian coordinates delNnn

 assert(LATTICED==3);
  double delNnn[nLines][LATTICED];
  for (int iLi=0;iLi<nLines;iLi++)
      {
      for (int i=0;i<LATTICED;i++)
          {
           delNnn[iLi][i] = 0.;
           for (int j=0;j<LATTICED;j++)
                delNnn[iLi][i] += primLattice[j][i]*nnn[iLi][j];
          }
      }

// Second pass:
// identify and record neighbors in each shell.
double Jsum, Jsum2;
for (int iTo=0;iTo<nTot;iTo++)
    {
     // Initialize neighbor shell sizes
     for (int i=0;i<nShells[iTo];i++) shellSize[iTo][i] = 0;

if (nShells[iTo]>maxNbr) for (int i=0;i<nShells[iTo];i++) cout << sqrt(shellRadius2[iTo][i])<< " " ;

     int** iNbr;
     double** JNbr;
     assert(nShells[iTo]<maxNbr);

     iNbr = new int* [nShells[iTo]];
     JNbr = new double* [nShells[iTo]];

     for (int i=0;i<nShells[iTo];i++)
         {
          iNbr[i] = new int [maxNbr];
          JNbr[i] = new double [maxNbr];
          for (int j=0;j<maxNbr;j++) JNbr[i][j] = 0.;
         }


     int iLiMatch;
     int iM[nLines];
     for (int i=0;i<nLines;i++) iM[i]=0;
     Jsum2 = 0.;

     const double tol = 1.e-6;
     // Loop over replica cells and atomic basis.
     // xecg is offset vector (vecg and wecg are intermediate quantities),
     // delrCryst is vector between 2 atoms
     double vecg[LATTICED], wecg[LATTICED], xecg[LATTICED], delrCryst[LATTICED];
     // loop over cells guaranteed to encompass rmax
     for (int i=-nRhomb[0];i<=nRhomb[0];i++)
         {
          for (int j=-nRhomb[1];j<=nRhomb[1];j++)
              {
               vscopy(LATTICED,double(i),lattice[0],wecg);
               vaxpy (LATTICED,double(j),lattice[1],wecg);
               for (int k=-nRhomb[2];k<=nRhomb[2];k++)
                   {
                    vscopy(LATTICED,1.,wecg,xecg);
                    vaxpy (LATTICED,double(k),lattice[2],xecg);
                    for (int jTo=0;jTo<nTot;jTo++)
                        {
                         vscopy(LATTICED,-1.,glCoord[iTo],delrCryst);
                         vaxpy(LATTICED,+1.,glCoord[jTo],delrCryst);
                         vaxpy(LATTICED,+1.,xecg,delrCryst);
                         double dist2 = mydot(delrCryst,delrCryst);

                         // Do nothing if dist2, distance^2 
                         // to pair site jTo, is outside cutoff
                         // or if dist2 indicates this is a self-pair
                         if (dist2<=rmax2+tol && dist2>1.e-6)
                            {
                             // Match the Magnet variables glCoord
                             // and i,j,k with KKR variables tau2 and nnn.
                             // Match dist2 with shellRadius2.
                             // Save neighbor index and J in shell lists.

                             myInsert2(iTo,jTo,
                                       glName[iTo],glCoord[iTo],glName[jTo],glCoord[jTo],delrCryst,
                                       i,j,k,dist2,rmax2,
                                       nShells[iTo],shellRadius2[iTo],
                                       shellSize[iTo],maxNbr,iNbr,JNbr,
                                       primLattice,primInv,nLines,name1,tau1,name2,tau2,
                                       del,rsh,nnn,delNnn,JmeV,
                                       tol,iLiMatch);
assert(iLiMatch>-1);
 if (iM[iLiMatch]!=0) cout << iTo << "  " << iM[iLiMatch] << " " << iLiMatch << " i i i " << endl;
    assert (iM[iLiMatch]==0);
    iM[iLiMatch] = 1;
                             Jsum2 += JmeV[iLiMatch];
                            }
                        }
                   }
              }
         }
//cout << iTo << " etoiwuhRRR  ";

     // Now shellSize[iTo][*] = size of nbr shells around iTo'th site
     // Record the neighbor ID #'s
     Jsum = 0.;
     for (int jSh=0;jSh<nShells[iTo];jSh++) 
         {
          shellNbr[iTo][jSh] = new int [shellSize[iTo][jSh]];
          J       [iTo][jSh] = new double [shellSize[iTo][jSh]];
          J       [iTo][0][0]= 0.;
          for (int kNb=0;kNb<shellSize[iTo][jSh];kNb++)
              {
               shellNbr[iTo][jSh][kNb] = iNbr[jSh][kNb];
               J       [iTo][jSh][kNb] = 2.*JNbr[jSh][kNb];
               Jsum += J[iTo][jSh][kNb];
              }
         }
     for (int i=0;i<nShells[iTo];i++)
         {
          delete [] iNbr[i];
          delete [] JNbr[i];
         }
     delete [] iNbr;
     delete [] JNbr;
//cout << iTo << " etoiwuhRRR\n";
    }

   for (int iTo=0;iTo<nTot;iTo++)
   cout << rmax2 << " " << iTo << "'th makeNbrList: nShells Jsum "<< nShells[iTo] << " " << Jsum <<  " " << Jsum2 << endl;
   cout << endl;

   cout << " # shells for atom 0: " << nShells[0] << endl;
   cout << " shell sizes: ";
   for (int iSh=0;iSh<nShells[0];iSh++) cout << shellSize[0][iSh] << " ";
   cout << endl << endl;

   double Jdum = 0.;
   cout << " J's?: ";
   for (int iSh=0;iSh<nShells[0];iSh++) 
       {Jdum+=J[0][iSh][0]*shellSize[0][iSh]; cout << J[0][iSh][0] << " ";}
   cout << endl;
   cout << Jdum << " faux sum of J's " << endl;

   return 0;
}

//**********************************************************************
int makeListShellRad2(double dist2, double rmax2, int& nsh, int maxsh, 
             double rSh2[], double tol)
{
 // Do nothing if dist2, distance^2 between pair, is outside cutoff.
 if (dist2>(rmax2+tol)) return 0;

 // Skip if dist2 is a previously known neighbor shell.
 for (int i=0;i<nsh;i++) if (fabs(dist2-rSh2[i])<tol) return 0;

 // Is there room to add a shell?
 if (nsh==maxsh)
    {
     for (int i=0;i<min(10,nsh);i++) cout << rSh2[i] << " ";
     cout << endl;
     for (int i=max(10,nsh-10);i<nsh;i++) cout << rSh2[i] << " ";
     cout << endl;
     cout << nsh << " " << maxsh << " " << rmax2 << " rSh2 list is full!\n";
     exit(9);
     }

 //Is dist2 closer than any known shells?
 for (int i=0;i<nsh;i++)
     {
      if (dist2 < rSh2[i] - tol)
         {
          // Shift list of shell radii to make space for new element.
          for (int j=min(nsh-1,maxsh-2);j>=i;j--) rSh2[j+1]=rSh2[j];
          // Insert new element.
          nsh++;
          rSh2[i] = dist2;
          // successful add
          return 1;
         }
     }
 
 // At this point, rSh2[nsh] <= dist2 <= rmax2
 // add element to end of list
 nsh++;
 rSh2[nsh-1]=dist2;
 return 1;
}

//**********************************************************************
int myInsert2(int iTo, int jTo,
              string glNameI, double glCoorI[3],
              string glNameJ, double glCoorJ[3], double delrCryst[3],
              int i0, int j0, int k0, double pairIJDist2, double rmax2,
              int nSh, double shellRadius2[],
              int nNbr[], int maxnbr,
              int** iNbr, double** JNbr, 
              double primLattice[][LATTICED],double primInv[][LATTICED], int nLines,
              string name1[], double tau1[][3], string name2[], double tau2[][3],
              double del[][3], double rsh[], int nnn[][3], double delNnn[][3], double JmeV[],
              double tol, int& iLiMatch)
{
 assert(maxnbr==maxNbr);

 iLiMatch = -1;

 //verify consistency of Crystal data: that delrCryst and pairIJDist2 agree
// assert(fabs(pairIJDist2-mydot(delrCryst,delrCryst))<1.e-10);


 // Find the shell in Magnet structure that matches pairIJDist2
 // Magnet shellRadius2 includes r=0 self-pair
 int iShMatch = -1;
 for (int iSh=0;iSh<nSh;iSh++)
     if (fabs(pairIJDist2-shellRadius2[iSh])<tol)
        {
         assert (iShMatch==-1);  // each atom pair matches exactly one shell in Magnet structure
         iShMatch = iSh;
        }
 assert(iShMatch>-1);  // stop on failure to find a shell matching this pair

 // update the number of elements in this shell and update the temporary list
 nNbr[iShMatch]++;
 iNbr[iShMatch][nNbr[iShMatch]-1]=jTo;
 assert(nNbr[iShMatch]<maxNbr);

 // Find the corresponding pair in the KKR list
 // in order to assign the correct corresponding KKR J.
 // 1) Translate iTo'th, jTo'th atoms back to the unit cell at 000.
 //    Note that this is translated to the inside of the primitive cell, not just the simulation cell.


// assert(pairIJDist2<=rmax2+tol);

 // skip self-pairs
// if (pairIJDist2<1.e-4) return 0;

 // Check all nLines KKR pairs.

 iLiMatch = -1;
 int iLi;
 for (iLi=0;iLi<nLines;iLi++)
     {
      // skip obvious mismatches
      if (   glNameI!=name1[iLi] 
          || glNameJ!=name2[iLi]) continue;
      if (   fabs(pairIJDist2-rsh[iLi]*rsh[iLi])/(pairIJDist2+1.e-6)>1.e-6) continue;

      // verify that del and rsh agree
//    if (fabs(rsh[iLi]*rsh[iLi]-mydot(del[iLi],del[iLi]))/(rsh[iLi]*rsh[iLi]+1.e-10)>1.e-5)
//        cout << rsh[iLi]*rsh[iLi] << " " << mydot(del[iLi],del[iLi]) << " "
//             << fabs(rsh[iLi]*rsh[iLi]-mydot(del[iLi],del[iLi])) << endl;
//    assert(fabs(rsh[iLi]*rsh[iLi]-mydot(del[iLi],del[iLi]))/(rsh[iLi]*rsh[iLi]+1.e-10)<1.e-5);

//    double delta1[LATTICED], delta2[LATTICED], delNnn[LATTICED];
/*    double delCent1[LATTICED], delCent2[LATTICED];
      vscopy(LATTICED,-1.,tau1[iLi],delta1);
      vaxpy(LATTICED,+1.,glCoorI,delta1);
      vscopy(LATTICED,-1.,tau2[iLi],delta2);
      vaxpy(LATTICED,+1.,glCoorJ,delta2);
      translate(primLattice,primInv,delta1,delCent1);
      translate(primLattice,primInv,delta2,delCent2);
*/

//    vscopy(LATTICED,-1.,delrCryst,delta3);
//    translate(primLattice,primInv,delta3,delta4);
//    vaxpy(LATTICED,1.,delrCryst,delta3);

/*
      for (int i=0;i<LATTICED;i++)
          {
           delNnn[i] = 0.;
           for (int j=0;j<LATTICED;j++)
                delNnn[i] += primLattice[j][i]*nnn[iLi][j];
          }
*/

 // look for best match of delta vectors
      double d0, d1, d2;
      d0 =fabs(delrCryst[0]+tau1[iLi][0]-tau2[iLi][0]-delNnn[iLi][0]);
      d1 =fabs(delrCryst[1]+tau1[iLi][1]-tau2[iLi][1]-delNnn[iLi][1]);
      d2 =fabs(delrCryst[2]+tau1[iLi][2]-tau2[iLi][2]-delNnn[iLi][2]);


      if (
             d0+d1+d2<1.e-6
//        && fabs(delCent1[0])<1.e-4
//        && fabs(delCent1[1])<1.e-4
//        && fabs(delCent1[2])<1.e-4
//        && fabs(delCent2[0])<1.e-4
//        && fabs(delCent2[1])<1.e-4
//        && fabs(delCent2[2])<1.e-4
         )
         {
          iLiMatch=iLi;
          break;
         }
     }

 // iLi==nLines means that the atom pair could not be matched with KKR data

 if (iLi==nLines && pairIJDist2>1.e-4)
 {
      cout << pairIJDist2 << " " << rmax2 << " pairIJDist2\n\n";
      double tauCent1[LATTICED], tauCent2[LATTICED];
      translate(primLattice,primInv,tau1[iLiMatch],tauCent1);
      translate(primLattice,primInv,tau2[iLiMatch],tauCent2);
      cout << tauCent1[0] << " @ " << tauCent1[1] << " " << tauCent1[2] << endl;
      cout << tauCent2[0] << " @ " << tauCent2[1] << " " << tauCent2[2] << endl<< endl;

      cout << glCoorJ[0] << " @ " << glCoorJ[1] << " " << glCoorJ[2] << endl;

      cout << endl;
      cout << iTo << " @ " << jTo << " " << pairIJDist2 << " " << rsh[nLines-1]*rsh[nLines-1] << endl;
      cout << iLiMatch << endl;
      cout << iTo << " @ " << jTo << " " << pairIJDist2 << " " << rsh[iLiMatch]*rsh[iLiMatch] << endl<< endl;

 if (iLi==nLines) for (int jLi=0;jLi<nLines;jLi++)
      {
       if (fabs(pairIJDist2-rsh[jLi]*rsh[jLi])<1.e-2)
          {
          cout << jLi << " @ " << tau1[jLi][0] << " "  << tau1[jLi][1] << " "  << tau1[jLi][2];
          cout        << " @ " << tau2[jLi][0] << " "  << tau2[jLi][1] << " "  << tau2[jLi][2];
          cout << "   @   " << pairIJDist2 << " " << rsh[jLi]*rsh[jLi] << "    " << pairIJDist2-rsh[jLi]*rsh[jLi] << endl;
          }
      }
 }

 assert(iLi!=nLines || pairIJDist2<1.e-4); // stop on failure to find nondegenerate pair

 // Record the Heisenberg J for this pair.
 if (pairIJDist2>1.e-4) {JNbr[iShMatch][nNbr[iShMatch]-1] = JmeV[iLiMatch];}
 else  {JNbr[iShMatch][nNbr[iShMatch]-1] = 0.;}

assert(iLiMatch!=-1);

}
//**********************************************************************
int Magnet::secondPass(int nTot, int nLines, int glSpeciesID[], int glAtomID[],
                 int* nShells, double** shellRadius2,
                 int** shellSize, int*** shellNbr, double rmax)
{
 // This function is intended to work with the unit cell
 // used in computing Heisenberg J parameters.

 // Given the list of atoms of glSpeciesID, glAtomID
 // verify that neighbor lists for iTo'th atom make sense.
 // Read in a listing of the neighbors and compare to 
 // the results generated automatically.
 //
 // read in partial data in two passes.
 // first pass counted number of elements
 // second pass records pair atom indices,
 // relative separations, del*, (to low accuracy),
 // and (integer) replica cell displacement vectors, nnn

///////////////////////////////////////////////////
// make local copy of crystal data for comparison
 double lattice[3][3];
 int nSpecies;
 int* nAtoms;
 double ***basis;

 assert(LATTICED==3);

 for (int i=0;i<3;i++)
     xtlData.getIthLattice(i,lattice[i]);
    
 nSpecies = xtlData.getNumSpecies();
 nAtoms = new int [nSpecies];
 basis = new double**[nSpecies];
 for (int iSp=0;iSp<nSpecies;iSp++)
     {
      nAtoms[iSp] = xtlData.getNumAtoms(iSp);
      basis[iSp]  = new double* [nAtoms[iSp]];
      for (int jAt=0;jAt<nAtoms[iSp];jAt++)
          {
           basis[iSp][jAt] = new double [3];
           xtlData.getIJthBasis(iSp,jAt,basis[iSp][jAt]);
          }
     }

 // verify inverse of lattice matrix
 double a[3][3], b[3][3];
 inverse(lattice,a);
 transpose(a,b);
cout << lattice[0][0] << " " << lattice[0][1] << " " << lattice[0][2] << endl;
cout << b[0][0] << " " << b[0][1] << " " << b[0][2] << " LLL BBB " << endl;
 assert( fabs(dot(b[0],lattice[0])-1.) < 1.e-8);
 assert( fabs(dot(b[0],lattice[1])-0.) < 1.e-8);
 assert( fabs(dot(b[0],lattice[2])-0.) < 1.e-8);
 assert( fabs(dot(b[1],lattice[0])-0.) < 1.e-8);
 assert( fabs(dot(b[1],lattice[1])-1.) < 1.e-8);
 assert( fabs(dot(b[1],lattice[2])-0.) < 1.e-8);
 assert( fabs(dot(b[2],lattice[0])-0.) < 1.e-8);
 assert( fabs(dot(b[2],lattice[1])-0.) < 1.e-8);
 assert( fabs(dot(b[2],lattice[2])-1.) < 1.e-8);

///////////////////////////////////////////////////
// make a local copy of kkr data for comparison

 int indx1[nLines], indx2[nLines], ithshell[nLines],
     iinshell[nLines], glIndex[nLines], nnn[nLines][3];
 double rsh[nLines], JmRy[nLines], JmeV[nLines];
 double tau1[nLines][3], tau2[nLines][3], del[nLines][3];
 string name1[nLines], name2[nLines];

cout << nLines << endl;
 for (int i=0;i<nLines;i++)
     {
      indx1[i]    = kkr.getIndx1(i);
      indx2[i]    = kkr.getIndx1(i);
      ithshell[i] = kkr.getIthshell(i);
      iinshell[i] = kkr.getIinshell(i);
      glIndex[i]  = kkr.getGlIndex(i);
      rsh[i]      = kkr.getRsh(i);
      JmRy[i]      = kkr.getJmRy(i);
      JmeV[i]     = kkr.getJmeV(i);
      name1[i]    = kkr.getName1(i);
      name2[i]    = kkr.getName2(i);

      kkr.getNnn(i,nnn[i]);
      kkr.getTau1(i,tau1[i]);
      kkr.getTau2(i,tau2[i]);
      kkr.getDel (i,del [i]);
     }

 // Reconstruct pair displacement vectors.
 // The new values should be accurate to machine precision
 // where the input values were only 6-7 digits.
 double x, y, z;
 for (int i=0;i<nLines;i++)
     {
      x =-tau1[i][0] + tau2[i][0]
        + double(nnn[i][0])*lattice[0][0]
        + double(nnn[i][1])*lattice[1][0]
        + double(nnn[i][2])*lattice[2][0];
      y =-tau1[i][1] + tau2[i][1]
        + double(nnn[i][0])*lattice[0][1]
        + double(nnn[i][1])*lattice[1][1]
        + double(nnn[i][2])*lattice[2][1];
      z =-tau1[i][2] + tau2[i][2]
        + double(nnn[i][0])*lattice[0][2]
        + double(nnn[i][1])*lattice[1][2]
        + double(nnn[i][2])*lattice[2][2];
      if (   fabs(x-del[i][0])>1.e-4
          || fabs(y-del[i][1])>1.e-4
          || fabs(z-del[i][2])>1.e-4)
         {
          cout << tau1[i][0] << " " << tau1[i][1] << " " << tau1[i][2] << endl;
          cout << tau2[i][0] << " " << tau2[i][1] << " " << tau2[i][2] << endl;
          cout << nnn[i][0]  << " " << nnn[i][1]  << " " << nnn[i][2]  << endl;

          cout << i << " " << x << " uuuu " << del[i][0] << endl;
          cout << i << " " << y << " uuuu " << del[i][1] << endl;
          cout << i << " " << z << " uuuu " << del[i][2] << endl;

          cout << lattice[0][0] << " " << lattice[0][1] << " "  << lattice[0][2] << endl;
          cout << lattice[1][0] << " " << lattice[1][1] << " "  << lattice[1][2] << endl;
          cout << lattice[2][0] << " " << lattice[2][1] << " "  << lattice[2][2] << endl;
          assert(fabs(x-del[i][0])<1.e-4);
          assert(fabs(y-del[i][0])<1.e-4);
          assert(fabs(z-del[i][2])<1.e-4);
         }
      del[i][0] = x;
      del[i][1] = y;
      del[i][2] = z;
     }

// verify the atomic basis as derived from
// jrs.omni versus as generated in Crystal.

// 1) pick out each atom from kkr list
   int iatot = 0;
   for (int i=0;i<nLines;i++)
       {
        // new atom in kkr list
        if (iatot <indx1[i]) iatot = indx1[i];

        // Match every kkr atom with an atom in global list
        // Stop if failed to find.
        int j=0;
        while (   name1[i]!=glName[j] 
               && tau1[i][0]!=glCoord[j][0]
               && tau1[i][1]!=glCoord[j][1]
               && tau1[i][2]!=glCoord[j][2]
               ) {j++;if(j==nTot){cout<<"badgfd\n";exit(19);}}
       }
   // There must be the same # of atoms in kkr as in Crystal

   assert(nTot==iatot);

 // verify del, tau, and nnn for sigfig
 for (int i=0;i<nLines;i++)
     {
      double x[3], vecg[3];
      int index[3];

      x[0] = del[i][0] + tau1[i][0] - tau2[i][0];
      x[1] = del[i][1] + tau1[i][1] - tau2[i][1];
      x[2] = del[i][2] + tau1[i][2] - tau2[i][2];

      dot(b,x,vecg);

      index[0] = floor(vecg[0]+1.e-6);
      index[1] = floor(vecg[1]+1.e-6);
      index[2] = floor(vecg[2]+1.e-6);

      if (fabs(double(index[0])-vecg[0])>1.e-9)
         {cout << vecg[0] << " sig " << double(index[0])-vecg[0] << endl; exit(10);}
      if (fabs(double(index[1])-vecg[1])>1.e-9)
         {cout << vecg[1] << " sig " << double(index[1])-vecg[1] << endl; exit(10);}
      if (fabs(double(index[2])-vecg[2])>1.e-9)
         {cout << vecg[2] << " sig " << double(index[2])-vecg[2] << endl; exit(10);}

      // discrepancy?
      if (index[0]!=nnn[i][0] || index[1]!=nnn[i][1] || index[2]!=nnn[i][2])
         {
          cout << i << "  discrep    "
               << index[0] << " " << index[1] << " " << index[2] << "     "
               << nnn[i][0] << " " << nnn[i][1] << " " << nnn[i][2] << endl;
          exit(9);
         }

      x[0] =-tau1[i][0] + tau2[i][0]
        + double(nnn[i][0])*lattice[0][0]
        + double(nnn[i][1])*lattice[1][0]
        + double(nnn[i][2])*lattice[2][0];
      x[1] =-tau1[i][1] + tau2[i][1]
        + double(nnn[i][0])*lattice[0][1]
        + double(nnn[i][1])*lattice[1][1]
        + double(nnn[i][2])*lattice[2][1];
      x[2] =-tau1[i][2] + tau2[i][2]
        + double(nnn[i][0])*lattice[0][2]
        + double(nnn[i][1])*lattice[1][2]
        + double(nnn[i][2])*lattice[2][2];

      // CHECK old and new values
      // overwrite old values
      if    (   fabs(x[0]-del[i][0])>1.e-6
             || fabs(x[1]-del[i][1])>1.e-6
             || fabs(x[2]-del[i][2])>1.e-6)
            {
             cout << x[0] << " " << del[i][0] << endl;
             cout << x[1] << " " << del[i][1] << endl;
             cout << x[2] << " " << del[i][2] << endl;
             assert(   fabs(x[0]-del[i][0])<1.e-6
                    && fabs(x[1]-del[i][1])<1.e-6
                    && fabs(x[2]-del[i][2])<1.e-6);
            }
      del[i][0] = x[0];
      del[i][1] = x[1];
      del[i][2] = x[2];
      rsh[i] = sqrt(mydot(x,x));
     }

 // Find the maximum pair separation
 // This will control the minimum supercell size.
 for (int i=0;i<nLines;i++)
     {
      if (rsh[i]>rmax) rmax=rsh[i];
     }

}
