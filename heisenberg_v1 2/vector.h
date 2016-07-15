#include <cmath>

double det(const double a[3][3]);
double dot(const double a[3], const double b[3]);
void dot(const double a[3][3], const double b[3][3], double ab[3][3]);
void dot(const double a[3][3], const double b[3], double ab[3]);
void cross(const double a[3], const double b[3], double c[3]);
void transpose(const double a[3][3], double at[3][3]);
void inverse(const double a[3][3], double ai[3][3]);
int normalize(double a[3]);
int vscopy(int n, double a, double x[], double y[]);
int vaxpy(int n, double a, double x[], double y[]);
double mynorm(double v[]);
double mydot(double v[], double w[]);
