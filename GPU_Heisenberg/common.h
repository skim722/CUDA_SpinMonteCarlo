#ifndef __COMMON_H__

//Parallelism parameters.
#define THREADS_PER_BLOCK 64
#define LINE_LENGTH 4 //Number of elements each thread computes the spin on.

//Datatype parameters.
typedef float dtype;

//For random number generator.
#define RANDOM_A 1664525
#define RANDOM_B 1013904223
#include <limits>
#define NORMALIZE_RANDOM ( (dtype)1.0 / (dtype)std::numeric_limits<unsigned int>::max() )

typedef struct Spin2D
{
	dtype x,y;
} Spin2D;

typedef struct Dimension
{
	int x,y,z;
} Dimension;

typedef struct KernelDecomposition
{
	Dimension blocksPerGrid;
	Dimension threadsPerBlock;
} KernelDecomposition;

typedef struct KernelInputs
{
	dtype *x0,*y0,*z0,*x1,*y1,*z1;
	unsigned int* r;//random numbers (streams).
	Dimension L;//problem domain dimensions.
	dtype Hx,Hy,Hz;//H vector components.
	dtype delta_max;
	dtype T;//highest normalized temperature to descend from
	dtype J;
	Spin2D isingAxis;
	//Normalization of random number unsigned ints.
	dtype normRandom;
	//Derived variables.
	int nexty,maxx,maxy;
} KernelInputs;

#endif