
// Ising Model in two dimensions
// Generalized spins point in +/- some fixed direction in DIMENSIONS-d space

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <float.h>
#include <string>
#include <assert.h>

#include "heisenberg_old.h"

int findnRhomb(double rmax, double lattice[3][3], int nRhomb[3]);

using namespace std;

int none()
{


string cfile = "file.dat";
string mfile= "mfile.dat";
Magnet vry(cfile,mfile);

Magnet  mry;

double lattice[3][3];
vry.getIthLattice(0,lattice[0]);
vry.getIthLattice(1,lattice[1]);
vry.getIthLattice(2,lattice[2]);
double rmax;
rmax = vry.getRmax();

int nRhomb[3];
findnRhomb(rmax,lattice,nRhomb);

cout << " in Main " << rmax << " " << nRhomb[0] << " " << nRhomb[1] << " " << nRhomb[2] << endl;

// supercell should span -nRhomb to nRhomb, but the
// rhombus is already padded by one primitive cell,
// so estimate the size system needed to span rmax.
Magnet  dry(2*nRhomb[0],2*nRhomb[1],2*nRhomb[2],vry);

    Magnet  ery;

    ery=vry;
//exit(3);

//  assert (ery==vry);
}

int main (int argc, char *argv[]) {


  for (int i=0;i<10000;i++)
       none();

   srand(time(NULL)); 
   cout << " Two-dimensional Ising Model - Metropolis simulation\n"
          << " ---------------------------------------------------\n"
          << " Enter number of spins L in each direction(Box Size): ";
        
   cin >> Lx;
   Ly = Lx;
   N = Lx*Ly;

   double h;
   cout << " Enter magnetic field H: ";
   cin >> h;
   cout << " Enter number of Monte Carlo steps: ";
   int MCSteps;
   cin >> MCSteps;
    

   cout << " Enter the highest temperature T: ";
   cin >> T;
   //double T1=T;
   //double delta_T=T/1000;
   initialize();

   // define H vector
   for (int k=0;k<DIMENSIONS;k++) H[k] = h*isingAxis[k];

string cfile = "file.dat";
string mfile= "mfile.dat";
Magnet  vry(cfile,mfile);
   
   ofstream file("ising_quenching.data");
   file << " L(Box Size): " << Lx <<'\t' << " H(Magnetic Field): " << H <<'\t'<<" MCSteps: "<< MCSteps << endl;
   for (int t=0;t<500;t++){
     int thermSteps = int(0.2 * MCSteps);
     double mAv = 0, m2Av = 0, eAv = 0, e2Av = 0, MAv=0;
     cout<<"Starting Spin Set at T="<<T<<'\n'<<endl;
     for (int i = 0; i < Lx; i++){
       for (int j = 0; j < Ly; j++) {
       for (int k=0;k<DIMENSIONS;k++)
          cout<<spins[i][j][k]<<'\t';
          }
       cout<<'\n'<<endl;
       }
     
     for (int i = 0; i < thermSteps; i++){
          oneMonteCarloStepPerSpin();
          }
      int ispin = 0;
      double H[3];
      double SpinI[3];
       H[0] =0.;
       H[1] =0.;
       H[2] =0.;
     for (int k = 0; k < MCSteps; k++) {
          oneMonteCarloStepPerSpin();
          double m = fabs(magnetizationPerSpin(vry,H));
          double M = fabs(magnetization(vry,H));
          double e = energyPerSpin(ispin, vry, H);
          MAv +=M;
          mAv += m; m2Av += m * m;
          eAv += e; e2Av += e * e;
     }
     mAv /= MCSteps; m2Av /= MCSteps;
     eAv /= MCSteps; e2Av /= MCSteps;
     MAv /= MCSteps;
     
     cout<<"Final Spin Set at T="<<T<<'\n'<<endl;
     for (int i = 0; i < Lx; i++)
         {
          for (int j = 0; j < Ly; j++)
              {
               for (int k=0;k<DIMENSIONS;k++)
                    cout<<spins[i][j][k]<<'\t';
               cout << "  ";
              }
          cout<<'\n'<<endl;
         }
       cout <<"T ="<<'\t'<<T<< " <m> = "<<'\t' << mAv << " +/- " << sqrt(fabs(m2Av - mAv*mAv)) <<'\t'<<  " <e> = " <<'\t'<< eAv << " +/- " << sqrt(e2Av - eAv*eAv) << endl;
       file <<"T"<<'\t'<<  T <<'\t' << " <m> " <<'\t'<< mAv <<"   "<<'\t'<<"M"<<'\t'<<MAv<< endl;
       T=T-(T/1000)*(t);
       for (int i=0;i<Lx;i++){
       computeBoltzmannFactors2(i);
       }
}
  cout << " \n Magnetization and energy per spin written in file "
       << " \"ising_cpp.data\"" << endl;
   
  file.close();
}

