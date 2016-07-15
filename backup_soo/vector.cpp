#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "vector.h"
// det, (vector)dot, cross, transpose, (matrix)dot, inverse
////////////////////////////////////////////////////////////////////////
double det(const double a[3][3])
{
 return (
          a[0][0] * (a[1][1]*a[2][2] - a[1][2]*a[2][1])
        - a[0][1] * (a[1][0]*a[2][2] - a[1][2]*a[2][0])
        + a[0][2] * (a[1][0]*a[2][1] - a[1][1]*a[2][0])
        );
}
////////////////////////////////////////////////////////////////////////
double dot(const double a[3], const double b[3])
{
 return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
////////////////////////////////////////////////////////////////////////
void cross(const double a[3], const double b[3], double c[3])
{
c[0] = a[1]*b[2] - a[2]*b[1];
c[1] =-a[0]*b[2] + a[2]*b[0];
c[2] = a[0]*b[1] - a[1]*b[0];
}
////////////////////////////////////////////////////////////////////////
void transpose(const double a[3][3], double at[3][3])
{
 at[0][0] = a[0][0];
 at[0][1] = a[1][0];
 at[0][2] = a[2][0];

 at[1][0] = a[0][1];
 at[1][1] = a[1][1];
 at[1][2] = a[2][1];

 at[2][0] = a[0][2];
 at[2][1] = a[1][2];
 at[2][2] = a[2][2];
}
////////////////////////////////////////////////////////////////////////
void dot(const double a[3][3], const double b[3][3], double ab[3][3])
{
 double bt[3][3];

 transpose(b,bt);

 ab[0][0] = dot(a[0],bt[0]);
 ab[0][1] = dot(a[0],bt[1]);
 ab[0][2] = dot(a[0],bt[2]);

 ab[1][0] = dot(a[1],bt[0]);
 ab[1][1] = dot(a[1],bt[1]);
 ab[1][2] = dot(a[1],bt[2]);

 ab[2][0] = dot(a[2],bt[0]);
 ab[2][1] = dot(a[2],bt[1]);
 ab[2][2] = dot(a[2],bt[2]);
}
////////////////////////////////////////////////////////////////////////
void inverse(const double a[3][3], double ai[3][3])
{
 double ac[3][3];
 cross(a[1],a[2],ac[0]);
 cross(a[2],a[0],ac[1]); // minus sign
 cross(a[0],a[1],ac[2]);

 double di = 1./det(a);
 
 ac[0][0] *= di;
 ac[0][1] *= di;
 ac[0][2] *= di;
 ac[1][0] *= di;
 ac[1][1] *= di;
 ac[1][2] *= di;
 ac[2][0] *= di;
 ac[2][1] *= di;
 ac[2][2] *= di;

 transpose(ac,ai);
}
////////////////////////////////////////////////////////////////////////
int lattice2metric(const double a[3][3], double g[3][3])
{
 for (int i=0;i<3;i++)
     for (int j=0;j<3;j++)
         g[i][j] = dot(a[i],a[j]);
}
////////////////////////////////////////////////////////////////////////
int norm(double a[3]) {double r=1./sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
                       a[0]*=r;a[1]*=r;a[2]*=r;}
