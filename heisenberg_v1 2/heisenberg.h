#include <cmath>
#include <cstdlib>
#include <iostream>
//#include <boost/random/mersenne_twister.hpp>
//#include <boost/random/normal_distribution.hpp>
//#include <boost/random/variate_generator.hpp>
#include <fstream>
#include <float.h>
#include <string>
#include <assert.h>
#include <sstream>
#include "magnet.h"
#include "crystal.h"
#include "vector.h"

using namespace std;

inline double std_rand()
{
    return rand() / (RAND_MAX + 1.0);
}

//string cfile = "file.dat";
//string mfile= "mfile.dat";
//Magnet  max(cfile,mfile); // bring up file.dat mfile.dat and save into magnet object

//double J =1;                  // ferromagnetic coupling(1)
//int nTot;                     
//double **spins;                // the spins
//double T;                       // temperature
//double H[DIMENSIONS];           // magnetic field
//double isingAxis[DIMENSIONS];
//double delta_max; //HB metropolis step size
//long double w; //Boltzman factors

//Tools
inline double std_rand();
double mydot(double v[], double w[]);
double mynorm(double v[]);
float rand_gauss (void);
//Metropolis Monte Carlo 
void initialize(Magnet&);
bool MetropolisStep(Magnet& max, double delta_max);
void try_move(double * try_spin, double delta_max);
void try_move_sphere(double * try_spin);
void print_magnetic(string filename, Magnet& mag, double T);
int findnRhomb(double rmax, double lattice[3][3], int nRhomb[3]);
//Calculate Observable
double magnetizationPerSpin(Magnet&);
double magnetization(Magnet&);
double energyPerSpin(int i, Magnet& max);
double dEnergydSpin(int i, Magnet& max);
double acceptanceRatio;
int steps = 0;                  // steps so far


void initialize(Magnet& superMag) {

   double h;
   cout << " Enter magnetic field H: ";
   cin >> h;
   double H[3] = {0.,0.,1.};
   H[2] *= h;
 
   double T;
   cout << " Enter the temperature T: ";
   cin >> T;

   superMag.setH(H);
   superMag.setT(T);

   int nTot = superMag.getnTot();
   double spin[DIMENSIONS];
   for (int j = 0; j < nTot; j++)
       {
        // choose a random vector, uniformly distributed inside unit sphere
        // then normalize to a unit vector uniformly distributed over 4pi
        double s = 10.;
        while (s>1.0 || s==0.)
              {
               for (int i=0;i<DIMENSIONS;i++) spin[i] = 2.*std_rand()-1.0;
               s = mynorm(spin);
              }
        for (int k=0;k<DIMENSIONS;k++) spin[k] /= s;

        superMag.writeIthSpin(j,spin);
       }

     steps = 0;
     superMag.setnMonte(steps);
     print_magnetic("initial",superMag,T);
}

void initialize2(Magnet& superMag) {

  double h;
  cout << " Enter magnetic field H: ";
  cin >> h;
  double H[3] = {0.,0.,1.};
  H[2] *= h;

  double T;
  cout << " Enter the temperature T: ";
  cin >> T;

  superMag.setH(H);
  superMag.setT(T);
}

void initialize(string filename, Magnet& superMag) {

   ifstream fin;
   fin.open(filename.c_str());

   double T;
   fin >> T;
   superMag.setT(T);

   double H[3];
   fin >> H[0] >> H[1] >> H[2];
   superMag.setH(H);

   int nTot0;
   fin >> nTot0;
   int nTot = superMag.getnTot();
   assert (nTot==nTot0);

   int nMonte;
   fin >> nMonte;
   superMag.setnMonte(nMonte);


   double spin[DIMENSIONS];
   for (int j = 0; j < nTot; j++)
       {
        fin >> spin[0] >> spin[1] >> spin[2];
        superMag.writeIthSpin(j,spin);
       }

     steps = nMonte;
     print_magnetic("initial",superMag,T);
}

void print_magnetic(string filename, Magnet& mag, double T){
  ostringstream oss;
  oss<<T;
  filename=filename+"_"+oss.str()+".data";
  ofstream file(filename.c_str());
  int nTot = mag.getnTot();
  double spin[3];

  cout<<"Target T="<< T << endl;
      for (int j = 0; j < nTot; j++)
	{
          mag.getIthSpin(j,spin);
	  //  cout<<"\n";
	  file<<"\n";
	  for (int k=0;k<DIMENSIONS;k++)
	    {
	      //  cout<<spin[k]<<'\t';
	      file<<spin[k]<<" ";
	    }
	  // cout<<"\n";
	  file<<"\n";
	}
  file.close();

}

// looks OK 7/20/14 MPS
//Random Movement through the Sphere
void try_move(double * try_spin, double delta_max){

  double delta [DIMENSIONS];
    // choose a random vector, uniformly distribute inside unit sphere
    double s = 10.;
    while (s>1.0 || s==0.)
          {
           for (int i=0;i<DIMENSIONS;i++) delta[i] = 2.*std_rand()-1.0;
           s = mynorm(delta);
          }

    for (int k=0;k<DIMENSIONS;k++) delta[k] *= delta_max/s;
    for (int k=0;k<DIMENSIONS;k++) try_spin[k]=try_spin[k]+delta[k];
    s=mynorm(try_spin);
    for (int k=0;k<DIMENSIONS;k++) try_spin[k]/=s;

}


bool MetropolisStep (Magnet& max, double delta_max) {
  int nTot=max.getnTot(); 
  double T=max.getT();
//cout << T << " " ;
  double SpinI[3];
    
//cout << i << " random site " << endl;

     //Compute energy of initial configuration
     double e0 = dEnergydSpin( i,  max);
     
     //Random move to spin i,j
     double try_spin[3];

     // save old spin
     max.getIthSpin(i, SpinI);    
     for (int k=0;k<3;k++){try_spin[k]=SpinI[k];}

     // Move spin direction and overwrite our spin database
     try_move(try_spin, delta_max);
     max.writeIthSpin(i,try_spin);

     //Compute energy of trial configuration
     double e1=  dEnergydSpin( i,  max);

     //cout << e1-e0 << "  l  " << e1 << endl;

     // ratio of Boltzmann factors
     long double ratio = exp(-(e1-e0)/(T+1e-8));
     
     if (std_rand() < ratio) {
          for (int k=0;k<DIMENSIONS;k++) 
	    return true; // leave as the modified dataset
     } 
     else{
       max.writeIthSpin(i,SpinI); //return to the unmodified sataus
        return false;
     }
}


double oneMonteCarloStepPerSpin (Magnet& max, const double delta_max) {
     int accepts = 0;

     for (int i = 0; i < max.getnTot(); i++)
          if (MetropolisStep(max,delta_max))
               ++accepts;

     acceptanceRatio = double(accepts)/double(max.getnTot());
     ++steps;
     return acceptanceRatio;
}


double magnetizationPerSpin(Magnet& max){
  double spinI[3];
  int nTot=max.getnTot();
  
  double ssum[DIMENSIONS];
  for (int k=0;k<DIMENSIONS;k++) ssum[k]=0.0;
  for (int i=0;i<nTot;i++)
     {
        max.getIthSpin(i,spinI);
        for (int k=0;k<DIMENSIONS;k++)
           ssum[k] +=  spinI[k];
     }

  return mynorm(ssum)/double(nTot);
}

double magnetization(Magnet& max){
  double spinI[3];
  int nTot=max.getnTot();
  
  double ssum[DIMENSIONS];
  for (int k=0;k<DIMENSIONS;k++) ssum[k]=0.0;
  for (int i=0;i<nTot;i++)
     {
        max.getIthSpin(i,spinI);
        for (int k=0;k<DIMENSIONS;k++)
           ssum[k] +=  spinI[k];
     }

  return mynorm(ssum);
}

double energyPerSpin(int i, Magnet& max)
{
 // energy for i'th spin, 
 // Heisenberg terms are split 50/50 between site i and neighbor sites
 // Sum this function over all sites i to get the total energy.

  double spinI[3], spinJ[3];
  int nShellI = max.getIthNShell(i);
  int shellSizeI[nShellI];
  for (int j=0;j<nShellI;j++)
      shellSizeI[j] = max.getIJthShellSize(i,j);

  max.getIthSpin(i,spinI);
  
 double energyI = 0.;
 for (int j=0;j<nShellI;j++)
     {
      for (int k=0;k<shellSizeI[j];k++)
          {
           int m = max.getIJKthNbr(i,j,k);
           double J = max.getIJKthJ(i,j,k);
           max.getIthSpin(m,spinJ);
           energyI += J * mydot(spinI,spinJ);
          }
     }

 double H[3];
 max.getH(H);

 // Use KKR sign convention for Heisenberg J
 return -energyI/2. + mydot(H,spinI);
 
    }

double dEnergydSpin(int i, Magnet& max)
{
 // Change in energy relative to setting the i'th spin to zero.
 // Use this function to compute the energy difference for altering a single spin.

 double H[3];
 max.getH(H);

 double spinI[3], spinJ[3];

 int nShellI = max.getIthNShell(i);

 int shellSizeI[nShellI];
 for (int j=0;j<nShellI;j++)
     shellSizeI[j] = max.getIJthShellSize(i,j);

/*
 int** shellNbrI;
 shellNbrI = new int* [nShellI];
 for (int j=0;j<nShellI;j++)
     {
      shellNbrI[j] = new int [shellSizeI[j]];
      for (int k=0;k<shellSizeI[j];k++)
          shellNbrI[j][k] = max.getIJKthNbr(i,j,k);
     }
*/

 max.getIthSpin(i,spinI);
  
 double energyI = 0.;
 for (int j=0;j<nShellI;j++)
     {
      for (int k=0;k<shellSizeI[j];k++)
          {
           int m = max.getIJKthNbr(i,j,k);
           double J = max.getIJKthJ(i,j,k);
           max.getIthSpin(m,spinJ);
           energyI += J * mydot(spinI,spinJ);
          }
     }

 // Use KKR sign convention for Heisenberg J
 // Return the energy difference between having
 // spinI at site i versus having (0,0,0) at site i.

 return -energyI + mydot(H,spinI);
 
    }
