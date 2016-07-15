#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <float.h>
#include <string>
#include <assert.h>

using namespace std;

const int DIMENSIONS=10000;

int test(int n1, int* n2, double  *** &x)
{
cout << " A" << endl;
x = new double **[n1];
for (int i=0;i<n1;i++)
    {
     x[i] = new double*[n2[i]];
     for (int j=0;j<n2[i];j++)
         {
          x[i][j] = new double[DIMENSIONS];
          for (int k=0;k<DIMENSIONS;k++)
              x[i][j][k]=(double)rand()*1000.;
 cout << i << " " << j << " " << x[i][j][0] << endl;
         }
    }

cout << x << endl;
return 0;
}

int untest(int n1, int* n2, double *** &x)
{
cout << " A" << endl;
for (int i=0;i<n1;i++)
    {
cout << " B" << endl;
     for (int j=0;j<n2[i];j++)
         {
 cout << " C " << i << " " << j << " " << x[i][j][0] << endl;
         delete[] x[i][j];
         }
exit(1);
     delete[] x[i];
    }
    delete[] x;
return 0;
}

int main ()
{
const int n1 = 4;
int n2[n1] = {2,3,2,4};
double *** x;
for (int i=0;i<100000;i++) 
for (int j=0;j<100000;j++) 
    {
     test(n1,n2,x);

cout << x << endl;

     cout << " R" << endl;
     for (int i=0;i<n1;i++)
     for (int j=0;j<n2[i];j++)
          cout << " D " << i << " " << j << " " << x[i][j][0] << endl;

     untest(n1,n2,x);
    }
}
