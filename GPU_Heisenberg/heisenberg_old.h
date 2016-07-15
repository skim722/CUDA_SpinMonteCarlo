// Ising Model in two dimensions
// Generalized spins point in +/- some fixed direction in DIMENSIONS-d space

//#static const int DIMENSIONS=3;

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <float.h>
#include <string>
#include <assert.h>
#include "magnet.h"

using namespace std;

inline double std_rand()
{
    return rand() / (RAND_MAX + 1.0);
}

double J =1;                  // ferromagnetic coupling(1)
int Lx, Ly;                     // number of spins in x and y
int N;                          // number of spins
double ***spins;                // the spins
double T;                       // temperature
double H[DIMENSIONS];           // magnetic field
double isingAxis[DIMENSIONS];
double delta_max; //HB metropolis step size


//long double w[17][3];                // Boltzmann factors
long double w; //Boltzman factors

double energyPerSpin(int i, Magnet max, double H[3]);

void computeBoltzmannFactors2(int i){
 
  double emax=log(DBL_MAX);
  Magnet max;
  double exponent= -1.0*energyPerSpin(i, max, H)/(T+1e-8);
          if (fabs(exponent)<emax)  w = exp(exponent);
          if (    (exponent)>emax)  w = DBL_MAX;
          if (    (exponent)<-emax) w = 0.;

}

int steps = 0;                  // steps so far

void initialize ( ) {

   // dimension the spins
     spins = new double** [Lx];
     for (int i = 0; i < Lx; i++)
         {
          spins[i] = new double* [Ly];
          for (int j = 0; j < Ly; j++)
               spins[i][j] = new double [DIMENSIONS];
         }

    // choose a random vector, uniformly distributed inside unit sphere
    // then normalize to a unit vector uniformly distributed over 4pi
    // ALL SPINS, H fields, etc. are constrained to be parallel
    // to this axis.  The system will behave exactly like the Ising model,
    // but the spins are treated as real vectors.
    double s = 10.;
    while (s>1.0 || s==0.)
          {
           for (int i=0;i<DIMENSIONS;i++) isingAxis[i] = 2.*std_rand()-1.0;
           s = mynorm(isingAxis);
          }
    for (int k=0;k<DIMENSIONS;k++) isingAxis[k] /= s;

    cout << "all spins are constrained to point +/- along isingAxis: ";
    for (int k=0;k<DIMENSIONS;k++) cout  << isingAxis[k] << " " ;
    cout << endl;

    // initialize spins: all pointing along +/- the same direction, isingAxis
     for (int i = 0; i < Lx; i++)
          for (int j = 0; j < Ly; j++)
              {
               s = std_rand() < 0.5 ? +1. : -1.;   // random flip
               for (int k=0;k<DIMENSIONS;k++)
                    spins[i][j][k] = s*isingAxis[k];   // hot start
              }
     //computeBoltzmannFactors2();
     steps = 0;
}

//Random Movement through the Sphere
void try_move(double * try_spin){

  double delta [DIMENSIONS];
    // choose a random vector, uniformly distribute inside unit sphere
    double s = 10.;
    while (s>1.0 || s==0.)
          {
           for (int i=0;i<DIMENSIONS;i++) delta[i] = 2.*std_rand()-1.0;
           s = mynorm(delta);
          }

    for (int k=0;k<DIMENSIONS;k++) delta[k] *= delta_max;
    for (int k=0;k<DIMENSIONS;k++) try_spin[k]=try_spin[k]+delta[k];
    s=mynorm(try_spin);
    for (int k=0;k<DIMENSIONS;k++) try_spin[k]/=s;

}

bool MetropolisStep_old ( ) {

     // choose a random spin site
     int i = int(Lx*std_rand());
     int j = int(Ly*std_rand());

     // find its neighbors using periodic boundary conditions
     int iPrev = i == 0 ? Lx-1 : i-1;
     int iNext = i == Lx-1 ? 0 : i+1;
     int jPrev = j == 0 ? Ly-1 : j-1;
     int jNext = j == Ly-1 ? 0 : j+1; 

     double sumNeighbors[DIMENSIONS], ss;
     for (int k=0;k<DIMENSIONS;k++)
         sumNeighbors[k] = spins[iPrev][j][k] + spins[iNext][j][k] + spins[i][jPrev][k] + spins[i][jNext][k];
     double delta_ss = 2.*mydot(spins[i][j],sumNeighbors);
        
     //Compute energy of initial configuration
     double e0 = -J*mydot(spins[i][j],sumNeighbors)-mydot(H,spins[i][j]); //initial energy
     
     //Random move to spin i,j
     double try_spin [DIMENSIONS];
     for (int k=0;k<DIMENSIONS;k++)
         try_spin[k]=spins[i][j][k];     
    
     try_move(try_spin);

     //Compute energy of initial configuration
     double e1 = -J*mydot(try_spin,sumNeighbors)-mydot(H,try_spin); //trial energy energy
           
     // ratio of Boltzmann factors
     long double ratio = exp(-(e1-e0)/(T+1e-8));
     

     if (std_rand() < ratio) {
          for (int k=0;k<DIMENSIONS;k++) spins[i][j][k]=try_spin[k];
          return true;
     } else return false;
}

//////////////////////////////////////////////////////////////////////////////////////////
bool MetropolisStep ( ) {
  Magnet max;
  double SpinI[3];
  int i= int(Lx*std_rand()) ;
     //Compute energy of initial configuration
     double e0 = -.1*energyPerSpin( i,  max, H);
     
     //Random move to spin i,j
     double try_spin[3];
     max.getIthSpin(i, SpinI);    
     for (int k=0;k<3;k++){try_spin[k]=SpinI[k];}
     try_move(try_spin);
     //Write moved spin direction on our spin database
     max.writeIthSpin(i,try_spin);
     //Compute energy of initial configuration
     double e1=  -.1*energyPerSpin( i,  max,  H);
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


double acceptanceRatio;

void oneMonteCarloStepPerSpin ( ) {
     int accepts = 0;
     for (int i = 0; i < N; i++)
          if (MetropolisStep())
               ++accepts;
     acceptanceRatio = accepts/double(N);
     ++steps;
}

//double magnetizationPerSpin ( ) {
//     double sSum[DIMENSIONS];
//     for (int k = 0; k<DIMENSIONS; k++) sSum[k] = 0.;
//
//     for (int i = 0; i < Lx; i++)
//     for (int j = 0; j < Ly; j++)
//     for (int k = 0; k<DIMENSIONS; k++)
//          sSum[k] += spins[i][j][k];
//
//     return mynorm(sSum) / double(N);
//}

double magnetizationPerSpin(Magnet max, double H[3]){
  double spinI[3];
  int N=max.getnTot();
  
  double ssum[DIMENSIONS];
  for (int k=0;k<DIMENSIONS;k++) ssum[k]=0.0;
  for (int i=0;i<N;i++)
     {
        max.getIthSpin(i,spinI);
        for (int k=0;k<DIMENSIONS;k++)
           ssum[k] +=  spinI[k];
     }

  return mynorm(ssum)/double(N);
}

double magnetization(Magnet max,double H[3]){
  double spinI[3];
  int N=max.getnTot();
  
  double ssum[DIMENSIONS];
  for (int k=0;k<DIMENSIONS;k++) ssum[k]=0.0;
  for (int i=0;i<N;i++)
     {
        max.getIthSpin(i,spinI);
        for (int k=0;k<DIMENSIONS;k++)
           ssum[k] +=  spinI[k];
     }

  return mynorm(ssum);
}


double energyPerSpin(int i, Magnet max, double H[3])
{
  double spinI[3], spinJ[3];
  max.getIthSpin(i,spinI);
  int nShellI = max.getIthNShell(i);
  int shellSizeI[nShellI];
  for (int j=0;j<nShellI;j++)
      shellSizeI[j] = max.getIJthShellSize(i,j);

 int** shellNbrI;
  shellNbrI = new int* [nShellI];
  for (int j=0;j<nShellI;j++)
      {
       shellNbrI[j] = new int [shellSizeI[j]];
       for (int k=0;k<shellSizeI[j];k++)
           shellNbrI[j][k] = max.getIJKthNbr(i,j,k);
      }

  max.getIthSpin(i,spinI);
  
 double energyI = 0.;
 for (int j=0;j<nShellI;j++)
     {
      for (int k=0;shellSizeI[j];k++)
          {
           double J = max.getIJKthJ(i,j,k);
           max.getIthSpin(j,spinJ);
        
           energyI += J * dot(spinI,spinJ);
          }
     }

// no factor of 1/2 for Metropolis deltaE
 return energyI + dot(H,spinI);
 
    }
