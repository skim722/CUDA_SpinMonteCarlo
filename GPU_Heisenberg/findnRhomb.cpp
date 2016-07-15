#include <cmath>
#include <iostream>
#include "vector.h"

using namespace std;

////////////////////////////////////////////////////////////////////////
 int findnRhomb(double rmax, double lattice[3][3], int nRhomb[3])
{
 // Given a sphere of radius rmax, find how many unit cells I must sum
 // over to encompass that sphere.  nRhomb is in lattice coordinates.
 // It is implicitly assumed that all atoms in the basis will lie
 // within one lattice box of one another, but this is not verified.
 // This could be tightened up a bit.

 double a[3][3], b[3][3], Rhomb[3];

 if (mynorm(lattice[0])<1.e-4 || mynorm(lattice[1])<1.e-4 || mynorm(lattice[2])<1.e-4)
    {
     cout << "in nRhomb\n";
     cout << rmax << endl;
     cout << lattice[0][0] << " " << lattice[0][1] << " " << lattice[0][2] << endl;
     cout << lattice[1][0] << " " << lattice[1][1] << " " << lattice[1][2] << endl;
     cout << lattice[2][0] << " " << lattice[2][1] << " " << lattice[2][2] << endl;
    }

 // inverse of lattice is related to reciprocal lattice
 // transpose because storage of 3-vectors is backwards vs. matrix indices
 inverse(lattice,a);
 transpose(a,b);

//cout << endl;
//cout << b[0][0] << " " << b[0][1] << " " << b[0][2] << endl;
//cout << b[1][0] << " " << b[1][1] << " " << b[1][2] << endl;
//cout << b[2][0] << " " << b[2][1] << " " << b[2][2] << endl;
//cout << endl;


 // let a = normalized lattice vectors, b = normalized reciprocal lattice vectors
 for (int i=0;i<3;i++)
     {
      double aa = 1./sqrt(dot(lattice[i],lattice[i]));
      double bb = 1./sqrt(dot(b[i],b[i]));
      for (int j=0;j<3;j++)
          {
           a[i][j] = lattice[i][j]*aa;
           b[i][j] *= bb;
          }
     }

//cout << dot(a[0],b[0]) << endl;
//cout << dot(a[0],b[0]) << endl;

 // lattice vectors define a parallelipiped with 6 faces.
 // The vector normal to ith face is b_i ~ a_j cross a_k
 // for unit vectors a_i ~ ith lattice vector (where "~" means parallel).
 // I want to place faces of the parallelipipeds at ends of b-segments of length rmax.
 // This corresponds to the end of a-segments of length rmax/(a_i dot b_i).

// cout << " BB a.b " << dot(a[0],b[0]) << " " << dot(a[1],b[0]) << " " << dot(a[2],b[0]) << endl;
// cout << " BB a.b " << dot(a[0],b[1]) << " " << dot(a[1],b[1]) << " " << dot(a[2],b[1]) << endl;
// cout << " BB a.b " << dot(a[0],b[2]) << " " << dot(a[1],b[2]) << " " << dot(a[2],b[2]) << endl;

 Rhomb[0] = rmax/fabs(dot(lattice[0],b[0]));
 Rhomb[1] = rmax/fabs(dot(lattice[1],b[1]));
 Rhomb[2] = rmax/fabs(dot(lattice[2],b[2]));

//cout << Rhomb[0] << " " << Rhomb[1] << " " << Rhomb[2] << endl;

 // Add 1 box shell for roundoff, and 1 extra shell as validation test.
 // No sites in the 2nd added, outermost shell may fall within rmax.
 // This can be used as a validation check in the calling routine.
 nRhomb[0]= int(Rhomb[0])+2;
 nRhomb[1]= int(Rhomb[1])+2;
 nRhomb[2]= int(Rhomb[2])+2;

//cout << nRhomb[0] << " " << nRhomb[1] << " " << nRhomb[2] << endl;

 return 0;
}

