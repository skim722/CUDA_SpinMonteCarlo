/*****************************************************/
/* KKR.h -- Class definitions for J-coupling files   */
/*****************************************************/
#ifndef KKR_H
#define KKR_H
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>

using namespace std;

class KKR {
public:
  KKR();                    //Default constructor
  KKR(const KKR & ref);     //Copy constructor
  ~KKR();                       //Deletes memory allocated to member basis

  const KKR &operator =(const KKR &);  //Assignment operator
  bool operator ==(const KKR &); //Equality operator
  bool operator !=(const KKR &); //Inequality operator

  int    getNLines() const {return nLines;}
  double getRmax() const  {return rmax;}
  int    getIndx1(int i) const {return indx1[i];}
  int    getIndx2(int i) const {return indx2[i];}
  int    getIthshell(int i) const {return ithshell[i];}
  int    getIinshell(int i) const {return iinshell[i];}
  int    getGlIndex(int i) const {return glIndex[i];}
  double getRsh(int i) const {return rsh[i];}
  double getJmRy(int i) const {return JmRy[i];}
  double getJmeV(int i) const {return JmeV[i];}
  string getName1(int i) const {return name1[i];}
  string getName2(int i) const {return name2[i];}
  int getNnn(int i, int n[3]) const {n[0]=nnn[i][0];n[1]=nnn[i][1];n[2]=nnn[i][2];}

  int getTau1(int i, double t[3]) const {t[0]=tau1[i][0];t[1]=tau1[i][1];t[2]=tau1[i][2];}
  int getTau2(int i, double t[3]) const {t[0]=tau2[i][0];t[1]=tau2[i][1];t[2]=tau2[i][2];}
  int getDel (int i, double d[3]) const {d[0]=del [i][0];d[1]=del [i][1];d[2]=del [i][2];}


private:

 int nLines;
 double rmax;

 // storage for pair list from KKR code
 // indx1,2 labels atoms in the basis of the unit cell
 // ithshell labels neighbor shell for pair of atoms "1", "2"
 // 

 int* indx1, *indx2, *ithshell, *iinshell, *glIndex;
 int** nnn;
 double* rsh, *JmRy, *JmeV;
 string* name1, *name2;
 double** tau1, **tau2, **del;

 int firstPass();
 int secondPass();

  int deallocateMemory(); //Releases memory - also used if/when re-sizing

};

#endif //KKR_H
