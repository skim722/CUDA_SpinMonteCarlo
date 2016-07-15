/// Heisenberg model  in two dimensions

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
using namespace std;

inline double std_rand()
{
    return rand() / (RAND_MAX + 1.0);
}

double J =1;                  // ferromagnetic coupling(1)
int Lx, Ly;                     // number of spins in x and y
int N;                          // number of spins
int **s;                        // the spins
double T;                       // temperature
double H;                       // magnetic field
int DIMENSIONS=3;
double w[17][3];                // Boltzmann factors

void computeBoltzmannFactors ( ) {
     for (int i = -8; i <= 8; i += 4) {
          w[i + 8][0] = exp( - (i * J + 2 * H) / (T+1e-4));
          w[i + 8][2] = exp( - (i * J - 2 * H) / (T+1e-4));
     }
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

     computeBoltzmannFactors ();
     // overwrite old Boltzmann factors with better ones
     computeBoltzmannFactors2();
     steps = 0;
}


void initialize_ising ( ) {
     s = new int* [Lx];
     for (int i = 0; i < Lx; i++)
          s[i] = new int [Ly];
     for (int i = 0; i < Lx; i++)
          for (int j = 0; j < Ly; j++)
               s[i][j] = std_rand() < 0.5 ? +1 : -1;   // hot start
     computeBoltzmannFactors();
     steps = 0;
}


bool MetropolisStep ( ) {

     // choose a random spin
     int i = int(Lx*std_rand());
     int j = int(Ly*std_rand());

     // find its neighbors using periodic boundary conditions
     int iPrev = i == 0 ? Lx-1 : i-1;
     int iNext = i == Lx-1 ? 0 : i+1;
     int jPrev = j == 0 ? Ly-1 : j-1;
     int jNext = j == Ly-1 ? 0 : j+1;

     // find sum of neighbors
     int sumNeighbors = s[iPrev][j] + s[iNext][j] + s[i][jPrev] + s[i][jNext];
     int delta_ss = 2*s[i][j]*sumNeighbors;

     // ratio of Boltzmann factors
     double ratio = w[delta_ss+8][1+s[i][j]];
     if (std_rand() < ratio) {
          s[i][j] = -s[i][j];
          return true;
     } else return false;
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

double magnetizationPerSpin ( ) {
     int sSum = 0;
     for (int i = 0; i < Lx; i++)
     for (int j = 0; j < Ly; j++) {
          sSum += s[i][j];
     }
     return double(sSum) / double(N);
}

double magnetization( ) {
     int sSum = 0;
     for (int i = 0; i < Lx; i++)
     for (int j = 0; j < Ly; j++) {
          sSum += s[i][j];
     }
     return sSum ;
}

double energyPerSpin ( ) {
     int sSum = 0, ssSum = 0;
     for (int i = 0; i < Lx; i++){
      for (int j = 0; j < Ly; j++) {
          sSum += s[i][j];
          int iNext = i == Lx-1 ? 0 : i+1;
          int jNext = j == Ly-2 ? 0 : j+1;
          ssSum += s[i][j]*(s[iNext][j] + s[i][jNext]);
     }
     return -double((J*ssSum + H*sSum))/double(N);
  }
}

int main (int argc, char *argv[]) {
    srand(time(NULL));
    cout << " Two-dimensional Ising Model - Metropolis simulation\n"
          << " ---------------------------------------------------\n"
          << " Enter number of spins L in each direction(Box Size): ";
     cin >> Lx;
     Ly = Lx;
     N = Lx*Ly;
     cout << " Enter magnetic field H: ";
     cin >> H;
     cout << " Enter number of Monte Carlo steps: ";
     int MCSteps;
     cin >> MCSteps;
     ofstream file("heisenberg.data"); 
     file << " L(Box Size): " << Lx <<'\t' << " H(Magnetic Field): " << H <<'\t'<<" MCSteps: "<< MCSteps << endl;
for (int t=0;t<1000;t++){
//     T=0+t*10;
     cout << " Enter temperature T: ";
     cin >> T;
     initialize();

     int thermSteps = int(0.2 * MCSteps);
//    cout << " Performing " << thermSteps
//         << " steps to thermalize the system ..." << flush;
     for (int i = 0; i < thermSteps; i++)
          oneMonteCarloStepPerSpin();

//     cout << " Done\n Performing production steps ..." << flush;
     double mAv = 0, m2Av = 0, eAv = 0, e2Av = 0, MAv=0;
//     ofstream file("ising_cpp.data");
     for (int k = 0; k < MCSteps; k++) {
          oneMonteCarloStepPerSpin();
          double m = fabs(magnetizationPerSpin());
          double M = fabs(magnetization());
          double e = energyPerSpin();
          MAv +=M;
          mAv += m; m2Av += m * m;
          eAv += e; e2Av += e * e;

     //  file <<"m :"<< m << '\t' <<"e :"<< e << '\n';
     }

    // file.close();
     mAv /= MCSteps; m2Av /= MCSteps;
     eAv /= MCSteps; e2Av /= MCSteps;
     MAv /= MCSteps;
     cout << " <m> = " << mAv << " +/- " << sqrt(fabs(m2Av - mAv*mAv)) <<'\t'<<  " <e> = " << eAv << " +/- " << sqrt(e2Av - eAv*eAv) << endl;
     cout<<"Spin :"<<'\n'<<endl;
     for (int i = 0; i < Lx; i++){
       for (int j = 0; j < Ly; j++) {
           cout<<s[i][j]<<'\t';
     }
     cout<<'\n'<<endl;
 }

     file <<"T"<<'\t'<<  T <<'\t' << " <m> " <<'\t'<< mAv <<"   "<<'\t'<<"M"<<'\t'<<MAv<< endl;
}
     file.close();
}
