
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <float.h>
#include <string>
#include <assert.h>
#include "crystal.h"
#include "magnet.h"
#include "heisenberg.h"

using namespace std;

int none()
{
Magnet  cry;
string cfile = "file.dat";
string mfile= "mfile.dat";
Magnet  vry(cfile,mfile);
Magnet  dry(cry);
Magnet  ery;
ery=vry;
assert (ery==vry);
}

int main (int argc, char *argv[]) {
   

   srand(time(NULL)); 
   cout << " Two-dimensional Heisenberg Model - Metropolis simulation\n"
          << " ---------------------------------------------------\n"
          << " Enter number of spins L in each direction(Box Size): ";
        
   cin >> Lx;
   Ly = Lx;
   Lz = Lx;
   N = Lx*Ly*Lz;

   double h;
   cout << " Enter magnetic field H: ";
   cin >> h;
   cout << " Enter number of Monte Carlo steps: ";
   int MCSteps;
   cin >> MCSteps;
    

   cout << " Enter the highest temperature T: ";
   cin >> T;
   initialize();

   // define H vector
   for (int l=0;l<DIMENSIONS;l++) H[l] = h*isingAxis[l];
   
   //Calculate delta_max
   double delta_max;
   delta_max = 0.1;
   //TODO: Need to fix. Seems optimization is not work. (Rooms for improvement) 
   for (int i=0;i<5000;i++){
          double ratio;
          ratio = oneMonteCarloStepPerSpin(delta_max);
           if     (ratio<0.30) {delta_max*=0.90;}
           else if(ratio<0.45) {delta_max*=0.99;}
           else if(ratio>0.70) {delta_max*=1.10;}
           else if(ratio>0.55) {delta_max*=1.01;}
          if (delta_max>0.5) {delta_max = 0.5;break;}
          cout << "T "<< T << " trial step size " << delta_max << " " << ratio << endl;
     }
   cout<<"optimized delta_max:  "<<delta_max<<endl;
   delta_max=0.15;                                        // TODO: delete later 
   ofstream file("heisenberg_quenching.data");
   file << " L(Box Size): " << Lx <<'\t' << " H(Magnetic Field): " << H <<'\t'<<" MCSteps: "<< MCSteps << endl;
   


for (int t=0;t<500;t++){
     int thermSteps = int(0.2 * MCSteps);
     double mAv = 0, m2Av = 0, eAv = 0, e2Av = 0, MAv=0;
//     cout<<"Starting Spin Set at T="<<T<<'\n'<<endl;
//     for (int i = 0; i < Lx; i++) {
//         for (int j = 0; j < Ly; j++) {
//             for (int k = 0; k < Lz; k++) {
//                 for (int l=0;l<DIMENSIONS;l++) {
//                     cout<<spins[i][j][k][l]<<'\t';
//                 }
//             }
//         }
//         cout<<'\n'<<endl;
//     }
     
     for (int i = 0; i < thermSteps; i++){
          oneMonteCarloStepPerSpin(delta_max);
     }

     for (int mc_n = 0; mc_n < MCSteps; mc_n++) {
          oneMonteCarloStepPerSpin(delta_max);
          double m = fabs(magnetizationPerSpin());
          double M = fabs(magnetization());
          double e = energyPerSpin();
          MAv +=M;
          mAv += m; m2Av += m * m;
          eAv += e; e2Av += e * e;
     }

     mAv /= MCSteps; m2Av /= MCSteps;
     eAv /= MCSteps; e2Av /= MCSteps;
     MAv /= MCSteps;
     
     cout<<"Final Spin Set at T="<<T<<'\n'<<endl;
     for (int i = 0; i < Lx; i++) {
          for (int j = 0; j < Ly; j++) {
               for (int k = 0; k < Lz; k++){
                   for (int l=0;l<DIMENSIONS;l++){
                             cout<<spins[i][j][k][l]<<'\t';
                   }
                   cout << "  ";
                }
                cout << '\n' << endl;;
            }
            cout<<'\n\n'<<endl;
       }
       cout <<"iteration number : "<<t<<"\n"<<endl;
       cout <<"T ="<<'\t'<<T<< " <m> = "<<'\t' << mAv << " +/- " << sqrt(fabs(m2Av - mAv*mAv)) <<'\t'<<  " <e> = " <<'\t'<< eAv << " +/- " << sqrt(e2Av - eAv*eAv) << endl;
       file <<"T"<<'\t'<<  T <<'\t' << " <m> " <<'\t'<< mAv <<"   "<<'\t'<<"M"<<'\t'<<MAv<< endl;
       T=T-(T/1000)*(t);
       computeBoltzmannFactors();
}

  cout << " \n Magnetization and energy per spin written in file "
       << " \"heisenberg_quencing.data\"" << endl;
   
  file.close();
}

