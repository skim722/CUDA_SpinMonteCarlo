
static const int DIMENSIONS=2;

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <float.h>
#include <string>
#include <assert.h>

using namespace std;

inline double std_rand()
{
    return rand() / (RAND_MAX + 1.0);
}

double J =1;                  // ferromagnetic coupling(1)
int Lx,Ly,Lz;                     // number of spins in x and y and z
int N;                          // number of spins
double ****spins;                // the spins
double T;                       // temperature
double H[DIMENSIONS];           // magnetic field
double isingAxis[DIMENSIONS];

long double w[17][3];                // Boltzmann factors

double mydot(double v[], double w[])
{
    double s = 0.;
    for (int l=0; l<DIMENSIONS; l++) s += v[l]*w[l];
    return s;
}
double mynorm(double v[]) { return sqrt(mydot(v,v)); }

void computeBoltzmannFactors ( ) {
     for (int i = -8; i <= 8; i += 4) {
          w[i + 8][0] = exp( - (i * J + 2 * mynorm(H)) / (T+1e-1));
          w[i + 8][2] = exp( - (i * J - 2 * mynorm(H)) / (T+1e-1));
          cout<<w[i+8][0]<<'\t'<<w[i+8][2]<<endl;
     }
}

void computeBoltzmannFactors2( ) {

     // modify code to prevent overflow, underflow
     double emax = log(DBL_MAX);

     for (int i = -8; i <= 8; i += 4) {

          double exponent =  - (i * J + 2 * mynorm(H)) / (T+1e-8);
          if (fabs(exponent)<emax)  w[i + 8][0] = exp(exponent);
          if (    (exponent)>emax)  w[i + 8][0] = DBL_MAX;
          if (    (exponent)<-emax) w[i + 8][0] = 0.;

          exponent = - (i * J - 2 * mynorm(H)) / (T+1e-8);
          if (fabs(exponent)<emax)  w[i + 8][2] = exp(exponent);
          if (    (exponent)>emax)  w[i + 8][2] = DBL_MAX;
          if (    (exponent)<-emax) w[i + 8][2] = 0.;

          cout << w[i+8][0] << " " << w[i+8][2] << endl;
     }
     cout << DBL_MAX << endl;
}


double dEnergydSpin(int i,int j,int k)
{

 // find its neighbors using periodic boundary conditions
 int iPrev = i == 0 ? Lx-1 : i-1;
 int iNext = i == Lx-1 ? 0 : i+1;
 int jPrev = j == 0 ? Ly-1 : j-1;
 int jNext = j == Ly-1 ? 0 : j+1;
 int kPrev = k == 0 ? Lz-1 : k-1;
 int kNext = k == Lz-1 ? 0 : k+1;
 double energyI = 0.;

 
 energyI += J*mydot(spins[i][j][k],spins[iPrev][j][k])+J*mydot(spins[i][j][k],spins[iNext][j][k])+J*mydot(spins[i][j][k],spins[i][jPrev][k])+J*mydot(spins[i][j][k],spins[i][jNext][k])+J*mydot(spins[i][j][k],spins[i][j][kPrev])+J*mydot(spins[i][j][k],spins[i][j][kNext]);
 return -energyI + 2*mydot(H,spins[i][j][k]);

    }




int steps = 0;                  // steps so far

void initialize ( ) {
     // dimension the spins
     spins = new double*** [Lx];
     for (int i = 0; i < Lx; i++) {
          spins[i] = new double** [Ly];
          for (int j = 0; j < Ly; j++) {
               spins[i][j] = new double* [Lz];
               for (int k = 0; k <Lz; k++){
                    spins[i][j][k] = new double [DIMENSIONS];
                }
           }
     }

    // choose a random vector, uniformly distributed inside unit sphere
    // then normalize to a unit vector uniformly distributed over 4pi
    // ALL SPINS, H fields, etc. are constrained to be parallel
    // to this axis.  The system will behave exactly like the Ising model,
    // but the spins are treated as real vectors.
    double s = 10.;
    while (s>1.0 || s==0.) {
           for (int l=0;l<DIMENSIONS;l++)
                 isingAxis[l] = 2.*std_rand()-1.0;
           s = mynorm(isingAxis);
    }
    for (int l=0;l<DIMENSIONS;l++) isingAxis[l] /= s;

    for (int l=0; l<DIMENSIONS; l++) cout  << isingAxis[l] << " " ;
    cout << endl;

    // initialize spins: all pointing along +/- the same direction, isingAxis
     for (int i = 0; i < Lx; i++)
          for (int j = 0; j < Ly; j++)
             for (int k = 0; k < Lz; k++) {
                  s = std_rand() < 0.5 ? +1. : -1.;   // random flip
                  for (int l=0;l<DIMENSIONS;l++)
                       spins[i][j][k][l] = s*isingAxis[l];   // hot start
              }

     computeBoltzmannFactors ();
     // overwrite old Boltzmann factors with better ones
     computeBoltzmannFactors2();
     steps = 0;
}

    
//Random Movement through the Sphere
double * try_move(double * try_spin, double delta_max){
    double delta [DIMENSIONS];
    // choose a random vector, uniformly distribute inside unit sphere
    double s = 10.;
    while (s>1.0 || s==0.) {
           for (int i=0;i<DIMENSIONS;i++) delta[i] = 2.*std_rand()-1.0;
           s = mynorm(delta);
    }

    for (int l=0;l<DIMENSIONS;l++) delta[l] *= delta_max/s;
    for (int l=0;l<DIMENSIONS;l++) try_spin[l]=try_spin[l]+delta[l];
    
    s=mynorm(try_spin);

    for (int l=0;l<DIMENSIONS;l++){
        try_spin[l]/=s;
    }


    return try_spin;
}

bool MetropolisStep (double delta_max, double T) {

     // choose a random spin site
     int i = int(Lx*std_rand());
     int j = int(Ly*std_rand());
     int k = int(Lz*std_rand());
     double e0=dEnergydSpin(i,j,k);
     double spins_before[DIMENSIONS];
     for (int l=0;l<DIMENSIONS;l++)
         spins_before[l]=spins[i][j][k][l];
    
     spins[i][j][k]=try_move(spins[i][j][k],delta_max);
     double e1=dEnergydSpin(i,j,k);
     //TODO
     long double ratio=exp(-(e1-e0)/(T+1e-8));

     // find its neighbors using periodic boundary conditions
     int iPrev = i == 0 ? Lx-1 : i-1;
     int iNext = i == Lx-1 ? 0 : i+1;
     int jPrev = j == 0 ? Ly-1 : j-1;
     int jNext = j == Ly-1 ? 0 : j+1;
     int kPrev = k == 0 ? Lz-1 : k-1;
     int kNext = k == Lz-1 ? 0 : k+1;

     double sumNeighbors[DIMENSIONS], ss;
     for (int l=0;l<DIMENSIONS;l++)
         sumNeighbors[l] = spins[iPrev][j][k][l] + spins[iNext][j][k][l] + spins[i][jPrev][k][l] + spins[i][jNext][k][l]+spins[i][j][kPrev][l]+spins[i][j][kNext][l];
     int delta_ss = rint(2.*mydot(spins[i][j][k],sumNeighbors));
     int mup = rint(mydot(spins[i][j][k],isingAxis)); // is spin up or down?

     // ratio of Boltzmann factors
     //long double ratio = w[delta_ss+8][1+mup];
     //cout<<"e1 "<<e1<<"e0 "<<e0<<endl;
     //cout<< "RATIO " <<ratio<<"\n";
     if (std_rand() < ratio) {
          for (int l=0;l<DIMENSIONS;l++) {
             return true;
          }
     } 
     else{
           for(int l=0;l<DIMENSIONS;l++)
                spins[i][j][k][l]=spins_before[l];
        return false;
     }
}



double acceptanceRatio;

double oneMonteCarloStepPerSpin (double delta_max, double T ) {
     int accepts = 0;
     for (int i = 0; i < N; i++)
          if (MetropolisStep(delta_max, T))
               ++accepts;
     acceptanceRatio = accepts/double(N);
     ++steps;
     return acceptanceRatio;
}

double magnetizationPerSpin ( ) {
     double sSum[DIMENSIONS];
     for (int l = 0; l<DIMENSIONS; l++) sSum[l] = 0.;

     for (int i = 0; i < Lx; i++)
       for (int j = 0; j < Ly; j++)
         for (int k = 0; k < Lz; k++)
           for (int l = 0; l<DIMENSIONS; l++)
              sSum[l] += spins[i][j][k][l];

     return mynorm(sSum) / double(N);
}

double magnetization( ) {
     double sSum[DIMENSIONS];
     for (int l = 0; l<DIMENSIONS; l++) sSum[l] = 0.;

     for (int i = 0; i < Lx; i++)
       for (int j = 0; j < Ly; j++)
         for (int k = 0; k < Lz; k++)
           for (int l = 0; l<DIMENSIONS; l++)
              sSum[l] += spins[i][j][k][l];
     return mynorm(sSum);
}

double energyPerSpin ( )
{
     double sSum[DIMENSIONS];
     for (int l = 0; l<DIMENSIONS; l++) sSum[l] = 0.;
     double ssSum = 0.;

     for (int i = 0; i < Lx; i++)
       for (int j = 0; j < Ly; j++)
          for (int k = 0; k < Lz; k++){
            int iNext = i == Lx-1 ? 0 : i+1;
            int jNext = j == Ly-1 ? 0 : j+1;
            int kNext = k == Lz-1 ? 0 : k+1;
            ssSum += mydot(spins[i][j][k],spins[iNext][j][k])+mydot(spins[i][j][k],spins[i][jNext][k])+mydot(spins[i][j][k],spins[i][j][kNext]);
            for (int l = 0; l<DIMENSIONS; l++)
                 sSum[l] += spins[i][j][k][l];
     }
     return -(J*ssSum + mydot(H,sSum))/N;
}
