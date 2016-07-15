#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <fstream>
#include <float.h>
#include <string>
#include <assert.h>
#include <sstream>
#include "heisenberg.h"
#include "vector.h"

using namespace std;

int main (int argc, char *argv[]) {

//part 1: Get data and construct magnetic object//
clock_t start;
 double duration_1, duration_2, duration_3, duration_4, duration_5, duration_6, duration_7,duration_8,duration_9, duration_10;
 start= clock();

// Construct manget object from constructed spin map 
int ispin = 0;
double H[3];
double SpinI[3];
H[0] =0.;
H[1] =0.;
H[2] =0.;

   srand(time(NULL)); 

   cout << " Automatically read in unit cell.\n";

   string cfile = "file.dat";
   string mfile= "mfile.dat";
   Magnet  unitMag(cfile,mfile);
   duration_1=(clock()-start)/(double)CLOCKS_PER_SEC;
   cout<<"Time consumed for part 1 :"<<duration_1<<'\n'; 
//part 2: Replicate supercell from unitcell//
   start=clock(); 
   cout << " Enter number of unit cells in each direction.\n";
        
   int n;
   cin >> n;
   
   cout << " Replicate supercell from unit cell.\n";
   Magnet superMag(n,n,n,unitMag);
   duration_2=(clock()-start)/(double)CLOCKS_PER_SEC;
   cout<<"Time consumed for part 2 :"<<duration_2<<'\n';
//part 3: Initialize Spin as the random fluctuaion and get input nTot, Stpes,T. //
   start=clock();

   int nTot = n*n*n*unitMag.getnTot();
   cout << " Total number of atoms is: " << nTot << endl;

   cout << " Enter number of Monte Carlo steps: ";
   int MCSteps;
   cin >> MCSteps;
   

//Initialize spin map
   cout<<"Is it initial MC run? OR do you want continue from previous run? \n";
   cout<<"(1) It is initial run for single temperature point \n";
   cout<<"(2) I want to contuinue from previous run. \n";
   int choi;
   cin>>choi;
   if (choi==1) initialize(superMag);
   else if (choi==2)initialize2(superMag);

   double T = superMag.getT();
   cout << " T " << T << endl;
   duration_3=(clock()-start)/(double)CLOCKS_PER_SEC;
   cout<<"Time consumed for part 3 :"<<duration_3<<'\n';
//part 4: Copy magnet object to dummag and optimize deltamax//                                         // copy Magnet object to temporary storage.                                                          
// Seek an optimal step size, delta_max.                                                             
// delta_max will depend on temperature.                                                            
// Then just use a fixed delta_max for the entire run.                                                                                                                                                
   start=clock();
   Magnet dumMag( superMag);
   double delta_max;
   delta_max = 0.1;
     for (int i=0;i<5000;i++)
         {
          double ratio;
          ratio = oneMonteCarloStepPerSpin(dumMag,delta_max);
	   if     (ratio<0.30) {delta_max*=0.90;}
           else if(ratio<0.45) {delta_max*=0.99;}
           else if(ratio>0.70) {delta_max*=1.10;}
           else if(ratio>0.55) {delta_max*=1.01;}
          if (delta_max>0.5) {delta_max = 0.5;break;}
	  cout << "T "<< T << " trial step size " << delta_max << " " << ratio << endl;
          }
     cout<<"delta_max  "<<delta_max<<endl; 
     duration_4=(clock()-start)/(double)CLOCKS_PER_SEC;
     cout<<"Time consumed for part 4 :"<<duration_4<<'\n';
//part 5: Get m_Fe and m_B to each spin//
//This is the step for accepting local m for multi component system 
double m_fe,m_b;
cout<<"This is Fe2-xCoxB calcaultion"<<'\n';
cout<<"Local effective m for Fe/Co :";
cin>>m_fe;
cout<<"Local m for B :";
cin>>m_b;

duration_5=(clock()-start)/(double)CLOCKS_PER_SEC;
 cout<<"Time consumed for part 5 :"<<duration_5<<'\n';
//part 6: All Monte Carlo Steps for MCStep sizes//
 start=clock();
//assert(9==3);

for (int t=0;t<1;t++){
 int thermSteps = int(0.2 * MCSteps);
 double mAv = 0, mAvFe=0, mAvB=0, m2Av = 0, eAv = 0, e2Av = 0, MAv=0; 
 cout<<"Starting Spin Set at T="<<T<<'\n'<<endl;
 //(o)Thermalization
 for (int i = 0; i < thermSteps; i++){
       oneMonteCarloStepPerSpin(superMag,delta_max);
       cout << i;
     }
 cout << endl;

 //assert(7<7);
 
 //(o)MonteCarlo Steps
 ostringstream temp;
 temp<<T;
 string name="output_";
 name=name+temp.str();
 ofstream file(name.c_str());
 file<<"=====(1)  Mangeitic moment and energy per each MC time steps ====="<<'\n';
for (int k0 = 0; k0 < MCSteps; k0++) {
          for (int m=0;m<100;m++) oneMonteCarloStepPerSpin(superMag,delta_max);

	  double ssum[3];
          double e = 0.;
	  double mfe,mb;
          double ssum_Fe[3],ssum_B[3];
          double spinI[3];
	  for (int k=0;k<3;k++) ssum_Fe[k]=ssum_B[k]=0.0;
          int nTot=superMag.getnTot();
          int n_Fe=0, n_B=0;
          for (int i=0;i<nTot;i++)
	    {   
                superMag.getIthSpin(i,spinI);      
             if (superMag.getglName(i)=="Fe"){
		for (int k=0;k<3;k++) ssum_Fe[k]+=m_fe*spinI[k];
		++n_Fe;
		  }
	     else if ( superMag.getglName(i)=="B"){
		for (int k=0;k<3;k++) ssum_B[k]+=m_b*spinI[k];
                ++n_B;
	          }
	      }
        
	  ssum[0]=ssum_B[0]+ssum_Fe[0]; 
          ssum[1]=ssum_B[1]+ssum_Fe[1];
          ssum[2]=ssum_B[2]+ssum_Fe[2];
 
          double m=mynorm(ssum)/double(nTot);
 
          mfe=mynorm(ssum_Fe)/double(n_Fe);
          mb=mynorm(ssum_B)/double(n_B);
          for (int n=0;n<nTot;n++) e+=energyPerSpin(n, superMag);
          
          mAv += m; m2Av += m * m;
          mAvFe+=mfe; mAvB+=mb;
          eAv += e; e2Av += e * e;
          
          file <<"T"<<'\t'<<  T <<'\t'<<"MC Steps"<<'\t'<< k0+1 <<'\t' << " m " <<'\t'<< m <<"   "<<'\t'<<" e = "<<'\t'<< e<<" sum of e "<<eAv<<" sum of e^2 " <<e2Av<< endl;	  
	  print_magnetic("initial",superMag,T);//write down spin maps so far for feeding it back to initial input at next iteration
     }


 mAv /= double(MCSteps); m2Av /= double(MCSteps); mAvFe /= double(MCSteps); mAvB /=double(MCSteps); eAv /=double(MCSteps); e2Av /=double(MCSteps); 

 double Cv=(e2Av-eAv*eAv)/(double(nTot)*T*T);
 file<<"=====(2) Total Magnetization after all "<<MCSteps<< " time stpes ======"<<'\n';
 file << "Ntot-: " << nTot <<'\t' << " H(Magnetic Field): " << H <<'\t'<<" MCSteps: "<< MCSteps <<'\n'<<"T"<<'\t'<<  T <<'\t' << " <m> " <<'\t'<< mAv <<'\t'<<"<m_Fe>"<<'\t'<<mAvFe<<"\t <m_B>"<<'\t'<<mAvB<<'\t'<<"M"<<'\t'<<MAv<<'\t'<<" <e> = " <<'\t'<< eAv<<'\t'<<"e2Av"<<"\t"<<e2Av<<" Cv "<<'\t'<<Cv<< endl; 

file.close();
   
 }
 duration_6=(clock()-start)/(double)CLOCKS_PER_SEC;
 cout<<"Time consumed for part 6 :"<<duration_6<<'\n';
                           
                                                                                                                               
                                                                                                                                                   
                                                                                                    
 
 //======================DONE and record time consumption=====================

 ofstream file("time_consumption.out");
 file<<"Part 1 : Get data and construct magnet object \n"<<duration_1<<'\n';
 file<<"Part 2 : Replicate supercell from unit cell \n"<<duration_2<<'\n';
 file<<"Part 3 : Initialize spin as a random fluctuation \n"<<duration_3<<'\n';
 file<<"Part 4 : Copy magnet object to dummag and optimize deltamax \n" <<duration_4<<'\n';
 file<<"Part 5 : Multiply m_Fe and m_B to each spins \n"<<duration_5<<'\n';
 file<<"Part 6 : All Monte Carlo Steps for MCStep sizes \n"<<duration_6<<endl;
 file.close();
}
