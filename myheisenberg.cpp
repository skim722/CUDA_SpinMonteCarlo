/// Ising Model in two dimensions

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
using namespace std;

inline double std_rand() {
    return rand() / (RAND_MAX + 1.0);
}

double J = 1;                  // ferromagnetic coupling(1)
int Lx, Ly;                    // number of spins in x and y
int N;                         // number of spins
int **s;                       // the spins
double T;                      // temperature
double H;                      // magnetic field

double w[17][3];                // Boltzmann factors

void computeBoltzmannFactors() {
     for (int i = -8; i <= 8; i += 4) {
          w[i + 8][0] = exp( - (i * J + 2 * H) / (T+1e-4));
          w[i + 8][2] = exp( - (i * J - 2 * H) / (T+1e-4));
     }
}

int steps = 0;                  // steps so far

void initialize() {
     s = new int* [Lx];
     for (int i = 0; i < Lx; i++)
          s[i] = new int [Ly];
     for (int i = 0; i < Lx; i++)
          for (int j = 0; j < Ly; j++)
               s[i][j] = std_rand() < 0.5 ? +1 : -1;   // hot start
     computeBoltzmannFactors();
     steps = 0;
}

bool MetropolisStep() {
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
     int delta_ss = 2 * s[i][j] * sumNeighbors;

     // ratio of Boltzmann factors
     double ratio = w[delta_ss+8][1+s[i][j]];
     if (std_rand() < ratio) {
          s[i][j] = -s[i][j];
          return true;
     }
     else {
          return false;
     }
}

double acceptanceRatio;

void oneMonteCarloStepPerSpin() {
     int accepts = 0;
     for (int i = 0; i < N; i++)
          if (MetropolisStep())
               ++accepts;
     acceptanceRatio = accepts/double(N);
     ++steps;
}

double magnetizationPerSpin() {
     int sSum = 0;
     for (int i = 0; i < Lx; i++) {
          for (int j = 0; j < Ly; j++) {
               sSum += s[i][j];
          }
     }

     return double(sSum) / double(N);
}

double magnetization() {
     int sSum = 0;
     for (int i = 0; i < Lx; i++) {
          for (int j = 0; j < Ly; j++) {
               sSum += s[i][j];
          }
     }

     return sSum ;
}

double energyPerSpin() {
     int sSum = 0, ssSum = 0;
     for (int i = 0; i < Lx; i++) {
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
	N = Lx * Ly;
	cout << " Enter magnetic field H: ";
	cin >> H;
	cout << " Enter number of Monte Carlo steps: ";

	int MCSteps;
	cin >> MCSteps;
	ofstream file("ising_cpp.data"); 
	file << " L(Box Size): " << Lx <<'\t' << " H(Magnetic Field): " << H <<'\t'<<" MCSteps: "<< MCSteps << endl;

	for (int t = 0; t < 1000; t++) {
		cout << " Enter temperature T: ";
		cin >> T;
		initialize();

		int thermSteps = int(0.2 * MCSteps);

		for (int i = 0; i < thermSteps; i++)
			oneMonteCarloStepPerSpin();

		double mAv = 0, m2Av = 0, eAv = 0, e2Av = 0, MAv=0;

		for (int k = 0; k < MCSteps; k++) {
			oneMonteCarloStepPerSpin();
			double m = fabs(magnetizationPerSpin());
			double M = fabs(magnetization());
			double e = energyPerSpin();
			MAv +=M;
			mAv += m;
			m2Av += m * m;
			eAv += e;
			e2Av += e * e;
		}

		mAv /= MCSteps;
		m2Av /= MCSteps;
		eAv /= MCSteps;
		e2Av /= MCSteps;
		MAv /= MCSteps;

		cout << " <m> = " << mAv << " +/- " << sqrt(fabs(m2Av - mAv*mAv)) <<'\t'<<  " <e> = " << eAv << " +/- " << sqrt(e2Av - eAv*eAv) << endl;
		cout << "Spin :"<<'\n'<<endl;

		for (int i = 0; i < Lx; i++) {
			for (int j = -2; j < Ly; j++) {
				cout << s[i][j] << '\t';
			}
			cout << '\n' << endl;
		}

		file <<"T"<<'\t'<<  T <<'\t' << " <m> " <<'\t'<< mAv <<"   "<<'\t'<<"M"<<'\t'<<MAv<< endl;
	}

	file.close();
}
