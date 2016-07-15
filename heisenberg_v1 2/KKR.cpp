/*******************************************************/
/* KKR.cpp -- Class definitions for J-coupling data.   */
/*******************************************************/
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include "KKR.h"
#include "vector.h"
#include <cmath>
#include <assert.h>

using namespace std;

int ReadLine(ifstream& fin,
             int& indx1, string& name1, int& indx2, string& name2,
             int& ithshell, int& iinshell, int& index, double del[],
             double& r, double& JmRy, double& JmeV, int nnn[]);

int ReadLine2(ifstream& fin,
             int& indx1, string& name1, int& indx2, string& name2,
             int& ithshell, int& iinshell, int& index, double del[],
             double& r, double& JmRy, double& JmeV,
             double tau1[], double tau2[]);

//*****************************************************************************
//Default constructor
KKR::KKR()
{

 nLines = 0;
 indx1 = indx2 = ithshell = iinshell = glIndex = NULL;
 nnn = NULL;
 rsh = JmRy = JmeV = NULL;
 name1 = name2 = NULL;
 tau1 = NULL;
 tau2 = NULL;
 del = NULL;
 rmax = 0.;


 // Read in nLines, rmax
 firstPass();

 assert(nLines>0);

 // allocate memory
 indx1    = new int [nLines];
 indx2    = new int [nLines];
 ithshell = new int [nLines];
 iinshell = new int [nLines];
 glIndex  = new int [nLines];
 rsh  = new double [nLines];
 JmRy  = new double [nLines];
 JmeV = new double [nLines];
 name1 = new string [nLines];
 name2 = new string [nLines];

 tau1 = new double* [nLines];
 tau2 = new double* [nLines];
 del  = new double* [nLines];
 nnn      = new int* [nLines];
 for (int i=0;i<nLines;i++)
     {
       nnn[i] = new int [3];
      tau1[i] = new double [3];
      tau2[i] = new double [3];
      del [i] = new double [3];
     }


 // Read in data
 secondPass();

 double Jsum = 0.;
 for (int i=0;i<nLines;i++) Jsum += JmeV[i];
 cout << " in KKR: total sum of J: " << Jsum << endl;

 // Now read in central cell coordinates and verify duplicate data from second file.
 ifstream fin;
 fin.open("jrs.omni");
 if (fin.fail()) {cout << " bad file jrs.omni\n";exit(9);}
 string str;
 getline(fin,str);
 assert(!fin.eof());
 for (int i=0;i<nLines;i++)
      ReadLine2(fin,
                indx1[i],name1[i],indx2[i],name2[i],ithshell[i],iinshell[i],glIndex[i],
                del[i],rsh[i],JmRy[i],JmeV[i],tau1[i],tau2[i]);

 fin.close();

}

//*****************************************************************************
//Copy constructor -- for instantiating objects with other objects
KKR::KKR(const KKR &refKKR)
{
 // copy all the KKR data 
 nLines = refKKR.nLines;

 rmax = refKKR.rmax;

 indx1    = new int [nLines];
 indx2    = new int [nLines];
 ithshell = new int [nLines];
 iinshell = new int [nLines];
 glIndex  = new int [nLines];
 rsh  = new double [nLines];
 JmRy  = new double [nLines];
 JmeV = new double [nLines];
 name1 = new string [nLines];
 name2 = new string [nLines];

 tau1 = new double* [nLines];
 tau2 = new double* [nLines];
 del  = new double* [nLines];
 nnn      = new int* [nLines];
 for (int i=0;i<nLines;i++)
     {
      nnn[i]  = new int [3];
      tau1[i] = new double [3];
      tau2[i] = new double [3];
      del [i] = new double [3];
     }

 for (int i=0;i<nLines;i++)
     {
      indx1[i]    = refKKR.indx1[i];
      indx2[i]    = refKKR.indx1[i];
      ithshell[i] = refKKR.ithshell[i];
      iinshell[i] = refKKR.iinshell[i];
      glIndex[i]  = refKKR.glIndex[i];
      rsh[i]      = refKKR.rsh[i];
      JmRy[i]      = refKKR.JmRy[i];
      JmeV[i]     = refKKR.JmeV[i];
      name1[i]    = refKKR.name1[i];
      name2[i]    = refKKR.name2[i];

      for (int j=0;j<3;j++)
          {
           nnn[i][j]  = refKKR.nnn[i][j];
           tau1[i][j] = refKKR.tau1[i][j];
           tau2[i][j] = refKKR.tau2[i][j];
           del [i][j] = refKKR.del[i][j];
          }
     }
 
}

//*****************************************************************************
//Default destructor
KKR::~KKR()
{
  deallocateMemory();
}

//*****************************************************************************
//Overload assignment operator
//  kkr1 = kkr2;
const KKR &KKR::operator =(const KKR &refKKR)
{
 //check for self-assignment and avoid
 if(&refKKR == this) return *this;

 if (nLines>0) deallocateMemory();

 nLines = refKKR.nLines;

 rmax = refKKR.rmax;

 indx1    = new int [nLines];
 indx2    = new int [nLines];
 ithshell = new int [nLines];
 iinshell = new int [nLines];
 glIndex  = new int [nLines];
 rsh  = new double [nLines];
 JmRy  = new double [nLines];
 JmeV = new double [nLines];
 name1 = new string [nLines];
 name2 = new string [nLines];

 tau1 = new double* [nLines];
 tau2 = new double* [nLines];
 del  = new double* [nLines];
 nnn      = new int* [nLines];
 for (int i=0;i<nLines;i++)
     {
       nnn[i] = new int [3];
      tau1[i] = new double [3];
      tau2[i] = new double [3];
      del [i] = new double [3];
     }

 for (int i=0;i<nLines;i++)
     {
      indx1[i]    = refKKR.indx1[i];
      indx2[i]    = refKKR.indx1[i];
      ithshell[i] = refKKR.ithshell[i];
      iinshell[i] = refKKR.iinshell[i];
      glIndex[i]  = refKKR.glIndex[i];
      rsh[i]      = refKKR.rsh[i];
      JmRy[i]      = refKKR.JmRy[i];
      JmeV[i]     = refKKR.JmeV[i];
      name1[i]    = refKKR.name1[i];
      name2[i]    = refKKR.name2[i];

      for (int j=0;j<3;j++)
          {
           nnn[i][j]  = refKKR.nnn[i][j];
           tau1[i][j] = refKKR.tau1[i][j];
           tau2[i][j] = refKKR.tau2[i][j];
           del [i][j] = refKKR.del[i][j];
          }
     }
cout << " KO5\n";

  return *this;  //enables x = y = z;
}

//*****************************************************************************
//Overload equality operator
bool KKR::operator ==(const KKR &refKKR)
{
 if (nLines!=refKKR.nLines) return false;

 if (rmax!=refKKR.rmax) return false;

 for (int i=0;i<nLines;i++)
     {
      if(indx1[i]   != refKKR.indx1[i])    return false;
      if(indx2[i]   != refKKR.indx1[i])    return false;
      if(ithshell[i]!= refKKR.ithshell[i]) return false;
      if(iinshell[i]!= refKKR.iinshell[i]) return false;
      if(glIndex[i] != refKKR.glIndex[i])  return false;
      if(rsh[i]     != refKKR.rsh[i])      return false;
      if(JmRy[i]    != refKKR.JmRy[i])     return false;
      if(JmeV[i]    != refKKR.JmeV[i])     return false;
      if(name1[i]   != refKKR.name1[i])    return false;
      if(name2[i]   != refKKR.name2[i])    return false;

      for (int j=0;j<3;j++)
          {
           if(nnn[i][j] != refKKR.nnn[i][j])  return false;
           if(tau1[i][j]!= refKKR.tau1[i][j]) return false;
           if(tau2[i][j]!= refKKR.tau2[i][j]) return false;
           if(del [i][j]!= refKKR.del[i][j])  return false;
          }
     }

 return true;

}

//*****************************************************************************
//Overload inequality operator
bool KKR::operator !=(const KKR &rhs)
{
  return !(*this == rhs);
}

//*****************************************************************************
//Deallocate Memory
int KKR::deallocateMemory()
{

 for (int i=0;i<nLines;i++)
     {
      delete [] tau1[i];
      delete [] tau2[i];
      delete [] del [i];
      delete [] nnn [i];
     }

 delete [] indx1;
 delete [] indx2;
 delete [] ithshell;
 delete [] iinshell;
 delete [] glIndex;
 delete [] rsh;
 delete [] JmRy;
 delete [] JmeV;
 delete [] name1;
 delete [] name2;

 delete [] tau1;
 delete [] tau2;
 delete [] del;
 delete [] nnn;

cout << "OO4\n";
}
//**********************************************************************
int KKR::firstPass()
{
 // read in partial KKR data in two passes.
 // first pass counts number of elements
 // second pass will record pair atom indices,
 // relative separations, del*, (to low accuracy),
 // and (integer) replica cell displacement vectors, nnn

 nLines = 0;

 // Open and read header line
 ifstream fin;
 fin.open("tmp_jrs.dat");
 if (fin.fail()) {cout << " bad file tmp_jrs.dat\n";exit(9);}
 int idum;
 double ddum;
 string str1, str2;
 getline(fin,str1);

 double r;
 while (!fin.eof())
       {
        fin >> idum >> str1 >> idum >> str2 >> idum >> idum >> idum
            >> ddum >> ddum >> ddum >> r >> ddum >> ddum >> ddum >> ddum >> ddum
            >> idum >> idum >> idum;
        if (r>rmax) rmax=r;
        if (!fin.eof()) nLines++;
       }
 fin.close();

}
//**********************************************************************
int KKR::secondPass()
{

 ifstream fin;
 string str;

////////////////////////////////////////////////
 // read in partial KKR data in two passes.
 // first pass counted number of elements
 // second pass records pair atom indices,
 // relative separations, del*, (to low accuracy),
 // and (integer) replica cell displacement vectors, nnn
 fin.open("tmp_jrs.dat");
 if (fin.fail()) {cout << " bad file tmp_jrs.dat\n";exit(9);}
 getline(fin,str);
 for (int i=0;i<nLines;i++)
     {
      ReadLine(fin,indx1[i],name1[i],indx2[i],name2[i],ithshell[i],iinshell[i],glIndex[i],
               del[i],rsh[i],JmRy[i],JmeV[i],nnn[i]);
     }
 fin.close();
}

//**********************************************************************
int ReadLine(ifstream& fin,
             int& indx1, string& name1, int& indx2, string& name2,
             int& ithshell, int& iinshell, int& glIndex,
             double del[], double& r,
             double& JmRy, double& JmeV, int nnn[])
{
 double dum;
 fin >> indx1 >> name1 >> indx2 >> name2 >> ithshell >> iinshell >> glIndex 
     >> del[0] >> del[1] >> del[2] >> r >> JmRy >> JmeV >> dum >> dum >> dum
     >> nnn[0] >> nnn[1] >> nnn[2];

 // verify that vector and scalar separations agree to reasonable precision
 if (fabs(r*r-mydot(del,del))/(r*r+1.e-10)>1.e-6)
    {
     cout << r*r << " FaiL " << mydot(del,del) << " " << r*r-mydot(del,del) << endl;
     assert(fabs(r*r-mydot(del,del))/(r*r+1.e-10)<1.e-6);
     // redefine scalar separation for machine precision
     r = sqrt(mydot(del,del));
    }


}

//**********************************************************************
int ReadLine2(ifstream& fin,
             int& indx1, string& name1, int& indx2, string& name2,
             int& ithshell, int& iinshell, int& glIndex,
             double del[], double& r,
             double& JmRy, double& JmeV,
             double tau1[], double tau2[])
{
 double dum, xr, xJmRy;
 double delx, dely, delz;
 int xindx1, xindx2;
 string xname1, xname2;

 if (fin.eof()) {cout << " discrepant length in jrs.omni file\n";exit(11);}

 fin >> xindx1 >> xname1 >> tau1[0] >> tau1[1] >> tau1[2]
     >> xindx2 >> xname2 >> tau2[0] >> tau2[1] >> tau2[2]
     >> delx   >> dely   >> delz   >> xr >> xJmRy;

 // Verify that this file agrees with the prior one.
 if (   xindx1!=indx1 || xname1!=name1 || xindx2!=indx2 || xname2!=name2
/*   || fabs(del[0]+delx+tau1[0]-tau2[0])>1.e-6
     || fabs(del[1]+dely+tau1[1]-tau2[1])>1.e-6
     || fabs(del[2]+delz+tau1[2]-tau2[2])>1.e-6*/)
    {
     cout << xindx1 << " " << indx1 << "  " << xindx2 << " " << indx2 << endl;
     cout << xname1 << " . " << name1 << " .  " << xname2 << " . " << name2 << endl;
     cout << del[0] << " " << delx  << "  " << del[1] << " " << dely  << " "
          << del[2] << " " << delz  << endl;
     cout << tau1[0] << " " << tau1[1] << " " << tau1[2] << endl;
     cout << tau2[0] << " " << tau2[1] << " " << tau2[2] << endl;
     cout << "diff " << del[0] -  delx  << "  " << del[1] - dely  << " " << del[2] - delz  << endl;
     cout << "Sum  " << del[0] +  delx  << "  " << del[1] + dely  << " " << del[2] + delz  << endl;
     cout << xr << " r " << r << endl;
     cout << xJmRy << " J " << JmRy/1000. << endl;
     cout << "file mismatch\n";
     exit(1);
    }
}
