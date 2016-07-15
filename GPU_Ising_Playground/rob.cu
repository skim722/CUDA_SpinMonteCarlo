/*
New Ising Kernel
CSE 6730 Project 2
*/
#include "cuda_utils.h"

typedef struct StopWatch
{
	//member variables
	cudaEvent_t m_startEvent,m_stopEvent;
	//constructor
	StopWatch()
	{
		cudaEventCreate(&m_startEvent);
		cudaEventCreate(&m_stopEvent);
	}
	//user functions
	void start()
	{
		cudaDeviceSynchronize();
		cudaEventRecord(m_startEvent, 0);
	}
	float stop()
	{
		cudaDeviceSynchronize();
		float elapsedTime;
		cudaEventRecord( m_stopEvent, 0 );
		cudaEventSynchronize( m_stopEvent );
		cudaEventElapsedTime( &elapsedTime, m_startEvent, m_stopEvent );
		return elapsedTime;
	}
	//destructor
	~StopWatch()
	{
		cudaEventDestroy( m_startEvent );
		cudaEventDestroy( m_stopEvent );
	}
} StopWatch;

bool validateKernelInt(int* d1,int* d2,int size)
{// Compares the int data in two gpu pointers. Should be the same. Returns false if not.
	int* h1 = new int[size];
	int* h2 = new int[size];

	CUDA_CHECK_ERROR(cudaMemcpy(h1,d1,size*sizeof(int),cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERROR(cudaMemcpy(h2,d2,size*sizeof(int),cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();

	for(int i=0;i<size;++i)
	{
		if(h1[i]!=h2[i])
		{
			printf("i,h1[i],h2[i]: %d,%d,%d\n",i,h1[i],h2[i]);
			printf("h1[i+1],h1[i+2]: %d,%d\n",h1[i+1],h1[i+2]);
			return false;
		}
	}

	delete[] h1;
	delete[] h2;

	return true;
}

bool validateCheckerboardInt(int* d1_0,int* d1_1,int* d2,int rows,int cols)
{// Compares the int data in three gpu pointers, two of which are segregated checkerboard data to be combined.
	//Should be the same. Returns false if not.
	int size = rows*cols;
	int checkerboardSize = size/2;
	int* h1_0 = new int[checkerboardSize];
	int* h1_1 = new int[checkerboardSize];
	int* h2 = new int[size];

	CUDA_CHECK_ERROR(cudaMemcpy(h1_0,d1_0,checkerboardSize*sizeof(int),cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERROR(cudaMemcpy(h1_1,d1_1,checkerboardSize*sizeof(int),cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERROR(cudaMemcpy(h2,d2,size*sizeof(int),cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();

	int checkerboardRows = rows;
	int checkerboardCols = cols/2;

	for(int row=0;row<checkerboardRows;++row)
	{
		for(int col=0;col<checkerboardCols;++col)
		{
			bool evenrow = row & 1;
			int a0 = 0;
			int a1 = 1;
			if(evenrow)
			{
				a0 = 1;
				a1 = 0;
			}
			int checkerboardIndex = row*checkerboardCols + col;
			if(
				h1_0[checkerboardIndex] != h2[checkerboardIndex*2 + a0]
				or
				h1_1[checkerboardIndex] != h2[checkerboardIndex*2 + a1]
			)
			{
				printf("i,h1_0,h1_1,h2[2*i],h2[2*i+1]: %d,%d,%d,%d,%d\n",
					checkerboardIndex,
					h1_0[checkerboardIndex],
					h1_1[checkerboardIndex],
					h2[2*checkerboardIndex+a0],
					h2[2*checkerboardIndex+a1]);
				return false;
			}
		}
	}

	delete[] h1_0;
	delete[] h1_1;
	delete[] h2;

	return true;
}


typedef struct KernelInputs
{
	int* S;//spins.
	int* R;//random numbers.
	int Lx,Ly;//problem dimensions.
	float exp4,exp8;//constants that serve as thresholds for new spin determination.
} KernelInputs;

#define NEXT_ROW (2*BLOCK_SIZE)

__global__ void device_rob(KernelInputs ki,bool alternate)
{
	int* S = ki.S;
	int* R = ki.R;
	bool flag = alternate;
	float exp_dH_4 = ki.exp4;
	float exp_dH_8 = ki.exp8;

	//Energy variable
	int dH=0;

	//Allocate shared memory
	int myRandom;

	//Load random data
	myRandom = R[threadIdx.x+BLOCK_SIZE*blockIdx.x];

	//Stencil computation on spins
	if(flag)
	{
		int spindex = 2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x;

		//Create new random numbers
		myRandom = RANDOM_A*myRandom+RANDOM_B;

		//Spin update top left
		bool block0 = blockIdx.x==0;
		int baseAdjust = block0 ? 0 : 4*BLOCK_SIZE*blockIdx.x;
		int base = 2*threadIdx.x+baseAdjust;
		int adjust = (threadIdx.x==0) ? 2*BLOCK_SIZE : 0;
		int adjust2 = block0 ? N : 0;
		dH=2*S[base]*(
			S[base+1]+//right
			S[base-1+adjust]+//left
			S[base+NEXT_ROW]+//bottom
			S[base-NEXT_ROW+adjust2]);//top

		bool useThreshold = dH==4 or dH==8;
		if(useThreshold)
		{
			float threshold = (dH==4) ? exp_dH_4 : exp_dH_8;
			if(fabs(myRandom*4.656612e-10)<threshold)
			{
				S[spindex]=-S[spindex];
			}
		}
		else
		{
			S[spindex]=-S[spindex];
		}

		//Create new random numbers
		myRandom=RANDOM_A*myRandom+RANDOM_B;

		base = 2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x;
		adjust = (threadIdx.x==BLOCK_SIZE-1) ? 0 : 2*BLOCK_SIZE;
		int left = (blockIdx.x==BLOCK_SIZE-1) ? (2*threadIdx.x+1) : (base+4*BLOCK_SIZE+1);
		//Spin update bottom right
		dH=2*S[base+1+2*BLOCK_SIZE]*(
			S[base+2+adjust]+
			S[base+2*BLOCK_SIZE]+
			S[left]+
			S[base+1]);

		spindex += 1+2*BLOCK_SIZE;
		useThreshold = dH==4 or dH==8;
		if(useThreshold)
		{
			float threshold = (dH==4) ? exp_dH_4 : exp_dH_8;
			if(fabs(myRandom*4.656612e-10)<threshold)
			{
				S[spindex]=-S[spindex];
			}
		}
		else
		{
			S[spindex]=-S[spindex];
		}

		//__syncthreads();
	}
	else
	{

		//Create new random numbers
		//r[threadIdx.x]=RANDOM_A*r[threadIdx.x]+RANDOM_B;
		myRandom = RANDOM_A*myRandom+RANDOM_B;

		//Spin update top right
		if(blockIdx.x==0)
		{ //Top
			if(threadIdx.x==BLOCK_SIZE-1)
			{ //Right
				dH=2*S[2*threadIdx.x+1]*(
					S[2*threadIdx.x+2-2*BLOCK_SIZE]+
					S[2*threadIdx.x]+
					S[2*threadIdx.x+1+2*BLOCK_SIZE]+
					S[2*threadIdx.x+1+N-2*BLOCK_SIZE]);
			}
			else
			{
				dH=2*S[2*threadIdx.x+1]*(
					S[2*threadIdx.x+2]+
					S[2*threadIdx.x]+
					S[2*threadIdx.x+1+2*BLOCK_SIZE]+
					S[2*threadIdx.x+1+N-2*BLOCK_SIZE]);
			}
		}
		else
		{
			if(threadIdx.x==BLOCK_SIZE-1)
			{ //Right
				dH=2*S[2*threadIdx.x+1+4*BLOCK_SIZE*blockIdx.x]*(
					S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2-2*BLOCK_SIZE]+
					S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x]+
					S[2*threadIdx.x+1+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE]+
					S[2*threadIdx.x+1+4*BLOCK_SIZE*blockIdx.x-2*BLOCK_SIZE]);
			}
			else
			{
				dH=2*S[2*threadIdx.x+1+4*BLOCK_SIZE*blockIdx.x]*(
					S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2]+
					S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x]+
					S[2*threadIdx.x+1+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE]+
					S[2*threadIdx.x+1+4*BLOCK_SIZE*blockIdx.x-2*BLOCK_SIZE]);
			}
		}

		if(dH==4)
		{
			//if(fabs(r[threadIdx.x]*4.656612e-10)<exp_dH_4)
			if(fabs(myRandom*4.656612e-10)<exp_dH_4)
			{
				S[2*threadIdx.x+1+4*BLOCK_SIZE*blockIdx.x]=-S[2*threadIdx.x+1+4*BLOCK_SIZE*blockIdx.x];
			}
		}
		else if(dH==8)
		{
			//if(fabs(r[threadIdx.x]*4.656612e-10)<exp_dH_8)
			if(fabs(myRandom*4.656612e-10)<exp_dH_8)
			{
				S[2*threadIdx.x+1+4*BLOCK_SIZE*blockIdx.x]=-S[2*threadIdx.x+1+4*BLOCK_SIZE*blockIdx.x];
			}
		}
		else
		{
			S[2*threadIdx.x+1+4*BLOCK_SIZE*blockIdx.x]=-S[2*threadIdx.x+1+4*BLOCK_SIZE*blockIdx.x];
		}

		//Create new random numbers
		//r[threadIdx.x]=RANDOM_A*r[threadIdx.x]+RANDOM_B;
		myRandom=RANDOM_A*myRandom+RANDOM_B;


		//Spin update bottom left
		if(blockIdx.x==BLOCK_SIZE-1)
		{ //Bottom
			if(threadIdx.x==0)
			{ //Left
				dH=2*S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE]*(
					S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE+1]+
					S[2*threadIdx.x+4*BLOCK_SIZE*(blockIdx.x+1)-1]+
					S[2*threadIdx.x]+
					S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x]);
			}
			else
			{
				dH=2*S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE]*(
					S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE+1]+
					S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE-1]+
					S[2*threadIdx.x]+
					S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x]);
			}
		}
		else
		{
			if(threadIdx.x==0)
			{ //Left
				dH=2*S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE]*(
					S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE+1]+
					S[2*threadIdx.x+4*BLOCK_SIZE*(blockIdx.x+1)-1]+
					S[2*threadIdx.x+4*BLOCK_SIZE*(blockIdx.x+1)]+
					S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x]);
			}
			else
			{
				dH=2*S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE]*(
					S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE+1]+
					S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE-1]+
					S[2*threadIdx.x+4*BLOCK_SIZE*(blockIdx.x+1)]+
					S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x]);
			}
		}

		if(dH==4)
		{
			//if(fabs(r[threadIdx.x]*4.656612e-10)<exp_dH_4)
			if(fabs(myRandom*4.656612e-10)<exp_dH_4)
			{
				S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE]=-S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE];
			}
		}
		else if(dH==8)
		{
			//if(fabs(r[threadIdx.x]*4.656612e-10)<exp_dH_8)
			if(fabs(myRandom*4.656612e-10)<exp_dH_8)
			{
				S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE]=-S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE];
			}
		}
		else
		{
			S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE]=-S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE];
		}
	}

	//Transfer random data back to global memory
	//R[threadIdx.x+BLOCK_SIZE*blockIdx.x]=r[threadIdx.x];
	R[threadIdx.x+BLOCK_SIZE*blockIdx.x]=myRandom;
}

typedef struct IsingKernelInputs2d
{
	int Lx,Ly;//problem dimensions. Should be even (multiple of 2)!
	int* S0;//checkerboard spin set 0. S[x_set0+y*Lx/2]
	int* S1;//checkerboard spin set 1.
	int* R;//random numbers.
	float exp4,exp8;//constants that serve as thresholds for new spin determination.
} IsingKernelInputs2d;

#define THREADS_PER_BLOCK 256 //must be even numbered!!!
#define LINE_LENGTH 16 //must be even numbered and multiple of Ly!!!

__device__ inline bool checkFlip(int dH,int randomValue,float exp4,float exp8)
{
	bool useThreshold = dH==4 or dH==8;
	float threshold = (dH==4) ? exp4 : exp8;
	bool meetThreshold = fabs(randomValue*4.656612e-10) < threshold;
	return ((not useThreshold) or (useThreshold and meetThreshold));
	// if(dH==4)
	// {
	// 	if(fabs(randomValue*4.656612e-10)<exp4)
	// 	{
	// 		return true;
	// 	}
	// }
	// else if(dH==8)
	// {
	// 	if(fabs(randomValue*4.656612e-10)<exp8)
	// 	{
	// 		return true;
	// 	}
	// }
	// else
	// {
	// 	return true;
	// }
	// return false;
}

__device__ inline int determineOffset(int i,int set0,int globalx,int maxx)
{//i is row number in line (0 to LINE_LENGTH-1 inclusive)
	// int oddrow = i & 1;
	// int offset = oddrow ? (set0 ? 1 : -1) : (set0 ? -1 : 1);
	// bool set1 = not set0;
	// bool evenrow = not oddrow;
	// bool needsLeft = (set0 and evenrow) or (set1 and oddrow);
	// bool needsRight = not needsLeft;
	// offset = (globalx==0 and needsLeft) ? maxx : offset;
	// offset = (globalx==maxx and needsRight) ? -maxx : offset;
	// return offset;
	int offset = (i & 1) ? (set0 ? 1 : -1) : (set0 ? -1 : 1);
	bool needsLeft = (set0 and (not (i & 1))) or ((not set0) and (i & 1));
	offset = (globalx==0 and needsLeft) ? maxx : offset;
	offset = (globalx==maxx and (not needsLeft)) ? -maxx : offset;
	return offset;
}

__global__ void ising2d(IsingKernelInputs2d ki,int* S,int* otherS,bool set0)
{//Segregated checkerboard.
	//Periodic BC.
	//Lx,Ly must be even-numbered!
	//LINE_LENGTH must be even numbered and multiple of Ly!!!
	//Coordinates hereafter refer to that of the segregated data (e.g. set 0, set 1)
	//Blocks are 1D, as each thread deals with columns of elements.
	//Grid is 2D.
	//System: Lx*Ly, Checkerboard: (Lx/2)*Ly, Random numbers: (Lx/2)*(Ly/LINE_LENGTH)

	//Let us check and see if the thread is in the computation domain.
	int globalx = blockDim.x*blockIdx.x + threadIdx.x;
	int globaly = blockIdx.y*LINE_LENGTH;
	int halfx = ki.Lx >> 1;
	int maxx = halfx - 1;
	int maxy = ki.Ly - 1;
	//if(globalx <= maxx and globaly <= maxy)//Let's compute only if we are in the domain.
	{
		//Let us see which checkerboard set we are dealing with.
		// int* S = set0 ? ki.S0 : ki.S1;
		// int* otherS = set0 ? ki.S1 : ki.S0;
		//convenience variables.
		int nexty = halfx;//next y row.
		int globalIndex = globalx+globaly*nexty;//serialized global index.
		int currentIndex,randomIndex;//indexing
		int change,top,center,off,bottom;//spins
		int randomValue,dH;//evaluation
		//intialize
		currentIndex = globalIndex;
		randomIndex = globalx+blockIdx.y*nexty;//(globaly>>1)*nexty;
		//First stencil!
		//Spins
		change = S[currentIndex];
		top = otherS[ (blockIdx.y==0) ? (currentIndex + maxy*nexty) : (currentIndex - nexty) ];
		center = otherS[currentIndex];
		off = otherS[ currentIndex+determineOffset(0,set0,globalx,maxx) ];
		bottom = otherS[ currentIndex+nexty ];
		//Evaluation metrics
		dH = (change<<1) * ( top + center + off + bottom );
		randomValue = RANDOM_A*ki.R[randomIndex]+RANDOM_B;
		if(checkFlip(dH,randomValue,ki.exp4,ki.exp8)) S[currentIndex]=-change;
		// S[currentIndex] = (checkFlip(dH,randomValue,ki.exp4,ki.exp8)) ? -S[currentIndex] : S[currentIndex];
		//End first stencil, start inner stencils
		for(int i=1;i<LINE_LENGTH-1;++i)
		{
			//increments
			randomValue = RANDOM_A*randomValue+RANDOM_B;
			// if(i & 1)
			// {
			// 	randomValue = RANDOM_A*randomValue+RANDOM_B;
			// 	ki.R[randomIndex] = randomValue;
			// }
			// else
			// {
			// 	randomIndex += nexty;
			// 	randomValue = RANDOM_A*ki.R[randomIndex]+RANDOM_B;
			// }
			currentIndex += nexty;
			change = S[currentIndex];
			top = center;
			center = bottom;
			off = otherS[ currentIndex+determineOffset(i,set0,globalx,maxx) ];
			bottom = otherS[ currentIndex+nexty ];
			dH = (change<<1) * ( top + center + off + bottom );
			if(checkFlip(dH,randomValue,ki.exp4,ki.exp8)) S[currentIndex]=-change;
			// S[currentIndex] = (checkFlip(dH,randomValue,ki.exp4,ki.exp8)) ? -S[currentIndex] : S[currentIndex];
		}
		//Last stencil!
		randomValue = RANDOM_A*randomValue+RANDOM_B;
		// ki.R[randomIndex] = randomValue;
		currentIndex += nexty;
		change = S[currentIndex];
		top = center;
		center = bottom;
		off = otherS[ currentIndex+determineOffset((LINE_LENGTH-1),set0,globalx,maxx) ];
		bottom = otherS[ (blockIdx.y==(gridDim.y-1)) ? globalx : (currentIndex+nexty) ];
		dH = (change<<1) * ( top + center + off + bottom );
		if(checkFlip(dH,randomValue,ki.exp4,ki.exp8)) S[currentIndex]=-change;
		// S[currentIndex] = (checkFlip(dH,randomValue,ki.exp4,ki.exp8)) ? -S[currentIndex] : S[currentIndex];
		//End last stencil!
		ki.R[randomIndex] = randomValue;
	}
}

__global__ void ising2d_registerStorage(IsingKernelInputs2d ki,bool set0)
{//Segregated checkerboard.
	//Periodic BC.
	//Lx,Ly must be even-numbered!
	//LINE_LENGTH must be even numbered and multiple of Ly!!!
	//Coordinates hereafter refer to that of the segregated data (e.g. set 0, set 1)
	//Blocks are 1D, as each thread deals with columns of elements.
	//Grid is 2D.
	//System: Lx*Ly, Checkerboard: (Lx/2)*Ly, Random numbers: (Lx/2)*(Ly/2)

	//Let us check and see if the thread is in the computation domain.
	int globalx = blockDim.x*blockIdx.x + threadIdx.x;
	int globaly = blockDim.y*blockIdx.y*LINE_LENGTH;
	int halfx = ki.Lx >> 1;
	int maxx = halfx - 1;
	int maxy = ki.Ly - 1;
	if(globalx <= maxx and globaly <= maxy)//Let's compute only if we are in the domain.
	{
		//Let us see which checkerboard set we are dealing with.
		int* S = set0 ? ki.S0 : ki.S1;
		int* otherS = set0 ? ki.S1 : ki.S0;

		//convenience variables.
		int nexty = halfx;//next y row.
		int globalIndex = globalx+globaly*nexty;//serialized global index.
		// int debugIndex = 230;bool debugSet = set0;

		//Transfer from global memory.
		//Topmost and bottommost values from global memory.
		int topIndex = (blockIdx.y==0) ? (globalIndex + maxy*nexty) : (globalIndex - nexty);
		int topSpin = otherS[topIndex];
		int bottomIndex = (blockIdx.y==(gridDim.y-1)) ? globalx : (globalIndex + LINE_LENGTH*nexty);
		int bottomSpin = otherS[bottomIndex];
		// if(globalIndex+(LINE_LENGTH-1)*nexty == debugIndex and debugSet)
		// {
		// 	printf("Bottom index, bottom spin: %d, %d\n",bottomIndex,bottomSpin);
		// 	printf("BLock id y, grid dim y: %d, %d\n",blockIdx.y,(gridDim.y-1));
		// }

		//Random values from global memory.
		int randomValues[LINE_LENGTH>>1];
		#pragma unroll
		for(int i=0;i<LINE_LENGTH>>1;++i)
		{
			int randomy = (globaly>>1) + i;
			//if(randomy <= (maxy>>1)) 
				randomValues[i] = ki.R[globalx+randomy*nexty];
		}
		//Spins to be changed from global memory.
		// int changeSpins[LINE_LENGTH];//Spins to be changed.
		// #pragma unroll
		// for(int i=0;i<LINE_LENGTH;++i)
		// {
		// 	//if(globaly+i <= maxy) 
		// 		changeSpins[i] = S[globalIndex+i*nexty];
		// }
		//Side spins from global memory.
		int mainSpins[LINE_LENGTH];//Spins on the main line (the ones that overlap).
		// int offSpins[LINE_LENGTH];//Spins off of the main line (the ones that do not overlap).
		#pragma unroll
		for(int i=0;i<LINE_LENGTH;++i)
		{
			//if(globaly+i <= maxy)
			{
				// int oddrow = i & 1;
				// int offset = oddrow ? (set0 ? 1 : -1) : (set0 ? -1 : 1);
				// bool set1 = not set0;
				// bool evenrow = not oddrow;
				// bool needsLeft = (set0 and evenrow) or (set1 and oddrow);
				// bool needsRight = not needsLeft;
				// offset = (globalx==0 and needsLeft) ? maxx : offset;
				// offset = (globalx==maxx and needsRight) ? -maxx : offset;
				// offSpins[i] = otherS[globalIndex+i*nexty+offset];
				// offSpins[i] = otherS[globalIndex+i*nexty+determineOffset(i,set0,globalx,maxx)];
				mainSpins[i] = otherS[globalIndex+i*nexty];
				// if(globalIndex+i*nexty == debugIndex and debugSet)
				// {
				// 	printf("CUDA debug: %d, %d, %d\n",needsLeft,needsRight,offset);
				// }
			}
		}
		//End transfer from global memory.
		
		//Advance random values.
		#pragma unroll
		for(int i=0;i<LINE_LENGTH>>1;++i)
		{
			randomValues[i] = RANDOM_A*randomValues[i]+RANDOM_B;
		}

		//Now do the meat of the computations!
		//if(blockIdx.y < (gridDim.y-1))
		{
			//First stencil!
			int changeSpin = S[globalIndex];
			int offSpin = otherS[globalIndex+determineOffset(0,set0,globalx,maxx)];
			// int dH = (changeSpin<<1) * ( topSpin + mainSpins[0] + offSpins[0] + mainSpins[1] );
			int dH = (changeSpin<<1) * ( topSpin + mainSpins[0] + offSpin + mainSpins[1] );
			if(checkFlip(dH,randomValues[0],ki.exp4,ki.exp8)) S[globalIndex]=-changeSpin;
			// int dH = (changeSpins[0]<<1) * ( topSpin + mainSpins[0] + offSpins[0] + mainSpins[1] );
			// if(checkFlip(dH,randomValues[0],ki.exp4,ki.exp8)) S[globalIndex]=-changeSpins[0];
			//End first stencil!
			//#pragma unroll
			for(int i=1;i<LINE_LENGTH-1;++i)
			{
				int changeSpin = S[globalIndex+i*nexty];
				int offSpin = otherS[globalIndex+i*nexty+determineOffset(i,set0,globalx,maxx)];
				//dH = (changeSpins[i]<<1) * ( mainSpins[i-1] + mainSpins[i] + offSpins[i] + mainSpins[i+1] );
				// int dH = (changeSpin<<1) * ( mainSpins[i-1] + mainSpins[i] + offSpins[i] + mainSpins[i+1] );
				int dH = (changeSpin<<1) * ( mainSpins[i-1] + mainSpins[i] + offSpin + mainSpins[i+1] );
				int oddrow = i & 1;	
				int randomValue = oddrow ? (RANDOM_A*randomValues[i>>1]+RANDOM_B) : randomValues[i>>1];
				//if(checkFlip(dH,randomValue,ki.exp4,ki.exp8)) S[globalIndex+i*nexty]=-changeSpins[i];
				if(checkFlip(dH,randomValue,ki.exp4,ki.exp8)) S[globalIndex+i*nexty]=-changeSpin;
				if(oddrow) ki.R[globalx+((globaly+i)>>1)*nexty] = randomValue;
			}
			//Last stencil!
			changeSpin = S[globalIndex+(LINE_LENGTH-1)*nexty];
			offSpin = otherS[globalIndex+(LINE_LENGTH-1)*nexty+determineOffset(LINE_LENGTH-1,set0,globalx,maxx)];
			// dH = (changeSpins[LINE_LENGTH-1]<<1) * ( 
			dH = (changeSpin<<1) * ( 
				mainSpins[LINE_LENGTH-2] + 
				mainSpins[LINE_LENGTH-1] + 
				offSpin + 
				// offSpins[LINE_LENGTH-1] + 
				bottomSpin );
			// int i = LINE_LENGTH-1;
			// if(globalIndex+i*nexty == debugIndex and debugSet)
			// {
			// 	printf("CUDA debug: %d, %d, %d, %d, %d, %d\n",dH,changeSpins[LINE_LENGTH-1],mainSpins[i-1],mainSpins[i],offSpins[i],bottomSpin);
			// }
			int randomValue = RANDOM_A*randomValues[(LINE_LENGTH-1)>>1]+RANDOM_B;
			if(checkFlip(dH,randomValue,ki.exp4,ki.exp8)) S[globalIndex+(LINE_LENGTH-1)*nexty]=-S[globalIndex+(LINE_LENGTH-1)*nexty];
			//if(checkFlip(dH,randomValue,ki.exp4,ki.exp8)) S[globalIndex+(LINE_LENGTH-1)*nexty]=-changeSpins[LINE_LENGTH-1];
			ki.R[globalx+((globaly+LINE_LENGTH-1)>>1)*nexty] = randomValue;
			//End last stencil!
		}
		/*
		else
		{//bottom most blocks
			int linelength = maxy - blockIdx.y*LINE_LENGTH + 1;
			//First stencil!
			int dH = (changeSpins[0]<<1) * ( topSpin + mainSpins[0] + offSpins[0] + mainSpins[1] );
			if(checkFlip(dH,ki.exp4,ki.exp8,randomValues[0])) S[globalIndex]=-changeSpins[0];
			//End first stencil!
			//#pragma unroll
			for(int i=1;i<linelength-1;++i)
			{
				dH = (changeSpins[i]<<1) * ( mainSpins[i-1] + mainSpins[i] + offSpins[i] + mainSpins[i+1] );
				int oddrow = i & 1;	
				int randomValue = oddrow ? (RANDOM_A*randomValues[i>>1]+RANDOM_B) : randomValues[i>>1];
				if(checkFlip(dH,ki.exp4,ki.exp8,randomValue)) S[globalIndex+i*nexty]=-changeSpins[i];
				if(oddrow) ki.R[globalx+((globaly+i)>>1)*nexty] = randomValue;
			}
			//Last stencil!
			dH = (changeSpins[linelength-1]<<1) * ( 
				mainSpins[linelength-2] + 
				mainSpins[linelength-1] + 
				offSpins[linelength-1] + 
				bottomSpin );
			int randomValue = RANDOM_A*randomValues[(linelength-1)>>1]+RANDOM_B;
			if(checkFlip(dH,ki.exp4,ki.exp8,randomValue)) S[globalIndex+(linelength-1)*nexty]=-changeSpins[linelength-1];
			ki.R[globalx+((globaly+linelength-1)>>1)*nexty] = randomValue;
			//End last stencil!
		}
		*/
	}
}


#define BLOCKDIMX 16
#define BLOCKDIMY 16

__global__ void ising2d_shmem(IsingKernelInputs2d ki,bool set0)
{// Using 2D lattice. I have coded it so that it is open to 3D lattice.
	//We assume the data inputs and thread / grid dimensions are 2D / 3D, to take advantage of shared memory organization.
	//Periodic BC.
	//One spin calculation per thread.
	//Lx,Ly must be even-numbered!
	//Remember that the region covered is blockDim.x*2 x blockDim.y.
	//Coordinates hereafter refer to that of the segregated data (e.g. set 0, set 1)

	//Let us check and see if the thread is in the computation domain.
	int localx = threadIdx.x;
	int localy = threadIdx.y;
	int globalx = blockDim.x*blockIdx.x + localx;
	int globaly = blockDim.y*blockIdx.y + localy;
	int halfx = ki.Lx >> 1;
	int maxx = halfx - 1;
	int maxy = ki.Ly - 1;
	if(globalx <= maxx and globaly <= maxy)//Let's compute only if we are in the domain.
	{
		//Let us see which checkerboard set we are dealing with.
		int* S = set0 ? ki.S0 : ki.S1;
		int* otherS = set0 ? ki.S1 : ki.S0;
		int localIndex = localx+localy*BLOCKDIMX;
		
		//Allocate the shared memory buffer.
		__shared__ int localSpins[BLOCKDIMX+BLOCKDIMY*BLOCKDIMX];//We say +2 for the boundary spins.
		
		//Let us transfer global memory to the shared memory buffer.
		//Fill in the internals.
		//helper variables for indexing.
		int nexty = halfx;//next y row.
		int prevy = -halfx;//previous y row.
		int globalIndex = globalx+globaly*nexty;//serialized global index.
		//copy element from other checkerboard with the same position coordinates as current element.
		localSpins[localx+localy*BLOCKDIMX] = otherS[globalIndex];
		bool oddrow = globaly & 1;//shift depends on whether the current row (y-coordinate) is odd or even.
		int currentSpin = S[globalIndex];
		int randomValueIndex = globalx+(globaly>>1)*BLOCK_SIZE;
		int randomValue = ki.R[randomValueIndex];
		randomValue = RANDOM_A*randomValue+RANDOM_B;
		randomValue = oddrow ? (RANDOM_A*randomValue+RANDOM_B) : randomValue;
		if(oddrow) ki.R[randomValueIndex] = randomValue;

		bool leftEdge = localx==0;
		bool rightEdge = globalx==maxx or localx==(blockDim.x-1);
		bool topEdge = localy==0;
		bool bottomEdge = globaly==maxy or localy==(blockDim.y-1);

		__syncthreads();//synchronize across the block so that the shared memory can be used.

		int top,right,bottom,left;
		int xshift = oddrow ? (set0 ? 0 : -1) : (set0 ? -1 : 0);
		//Left
		if(leftEdge)
		{
			bool set1 = not set0;
			bool evenrow = not oddrow;
			bool needsLeft = ((set0 and evenrow) or (set1 and oddrow));
			if(needsLeft)
			{
				int leftIndex = (globalx==0) ? (globalIndex+nexty-1): (globalIndex-1);
				left = otherS[leftIndex];
			}
			else
			{
				left = localSpins[localIndex+xshift];
			}
		}
		else
		{
			left = localSpins[localIndex+xshift];
		}
		//Right
		if(rightEdge)
		{
			bool set1 = not set0;
			bool evenrow = not oddrow;
			bool needsRight = rightEdge and ((set0 and oddrow) or (set1 and evenrow));
			if(needsRight)
			{
				int rightIndex = (globalx==maxx) ? (globalIndex+prevy+1): (globalIndex+1);
				right = otherS[rightIndex];
			}
			else
			{
				right = localSpins[localIndex+1+xshift];
			}
		}
		else
		{
			right = localSpins[localIndex+1+xshift];
		}
		//Top	
		if(topEdge)
		{
			int topIndex = (globaly==0) ? (globalx+maxy*nexty): (globalIndex+prevy);
			top = otherS[topIndex];
		}
		else
		{
			top = localSpins[localIndex - BLOCKDIMX];
		}
		//Bottom
		if(bottomEdge)
		{
			int bottomIndex = (globaly==maxy) ? (globalx): (globalIndex+nexty);
			bottom = otherS[bottomIndex];
		}
		else
		{
			bottom = localSpins[localIndex + BLOCKDIMX];
		}

		//Now do the computation.
		int dH = (currentSpin<<1) * ( top + right + bottom + left );

		// bool useThreshold = dH==4 or dH==8;
		// float threshold = (dH==4) ? ki.exp4 : ki.exp8;
		// bool meetThreshold = fabs(randomValue*4.656612e-10) < threshold;
		// if ((not useThreshold) or (useThreshold and meetThreshold))
		// {
		// 	S[globalIndex]=-currentSpin;
		// }
		if(checkFlip(dH,randomValue,ki.exp4,ki.exp8)) S[globalIndex]=-currentSpin;
	}
}

__global__ void ising2d_old(IsingKernelInputs2d ki,bool set0)
{// Using 2D lattice. I have coded it so that it is open to 3D lattice.
	//We assume the data inputs and thread / grid dimensions are 2D / 3D, to take advantage of shared memory organization.
	//Periodic BC.
	//One spin calculation per thread.
	//Lx,Ly must be even-numbered!
	//Remember that the region covered is blockDim.x*2 x blockDim.y.
	//Coordinates hereafter refer to that of the segregated data (e.g. set 0, set 1)

	//Let us check and see if the thread is in the computation domain.
	int localx = threadIdx.x;
	int localy = threadIdx.y;
	int globalx = blockDim.x*blockIdx.x + localx;
	int globaly = blockDim.y*blockIdx.y + localy;
	int halfx = ki.Lx >> 1;
	int maxx = halfx - 1;
	int maxy = ki.Ly - 1;
	if(globalx <= maxx and globaly <= maxy)//Let's compute only if we are in the domain.
	{
		//Let us see which checkerboard set we are dealing with.
		int* S = set0 ? ki.S0 : ki.S1;
		int* otherS = set0 ? ki.S1 : ki.S0;
		//int* R = set0 ? ki.R0 : ki.R1;
		
		//Allocate the shared memory buffer.
		__shared__ int localSpins[BLOCKDIMX+2][BLOCKDIMY+2];//We say +2 for the boundary spins.
		
		//Let us transfer global memory to the shared memory buffer.
		//Fill in the internals.
		//We need to add one to localx,y for shmemx,y because of boundary stencils.
		int shmemx = localx + 1;//shared memory x index.
		int shmemy = localy + 1;//shared memory y index.
		//helper variables for indexing.
		int nexty = halfx;//next y row.
		int prevy = -halfx;//previous y row.
		int globalIndex = globalx+globaly*nexty;//serialized global index.
		//copy element from other checkerboard with the same position coordinates as current element.
		localSpins[shmemx][shmemy] = otherS[globalIndex];

		//Fill in the boundaries.
		//periodic BC boundary transfers.
		//left boundary
		if(globalx==0)
		{
			localSpins[0][shmemy] = otherS[globalIndex+nexty-1];
		}
		else if(localx==0)
		{
			localSpins[0][shmemy] = otherS[globalIndex-1];
		}
		//right boundary
		if(globalx==maxx)
		{
			localSpins[shmemx+1][shmemy] = otherS[globalIndex+prevy+1];
		}
		else if(localx==blockDim.x-1)
		{
			localSpins[shmemx+1][shmemy] = otherS[globalIndex+1];
		}
		//top boundary
		if(globaly==0)
		{
			localSpins[shmemx][0] = otherS[globalx+maxy*nexty];
		}
		else if(localy==0)
		{
			localSpins[shmemx][0] = otherS[globalIndex+prevy];
		}
		//bottom boundary
		if(globaly==maxy)
		{
			localSpins[shmemx][shmemy+1] = otherS[globalx];
		}
		else if(localy==blockDim.y-1)
		{
			localSpins[shmemx][shmemy+1] = otherS[globalIndex+nexty];
		}
		
		int currentSpin = S[globalIndex];

		__syncthreads();//synchronize across the block so that the shared memory can be used.

		//Now do the computation.
		bool oddrow = globaly & 1;//shift depends on whether the current row (y-coordinate) is odd or even.
		int xshift = oddrow ? (set0 ? 0 : -1) : (set0 ? -1 : 0);
		int top = localSpins[shmemx][shmemy-1];
		int right = localSpins[shmemx+1+xshift][shmemy];
		int bottom = localSpins[shmemx][shmemy+1];
		int left = localSpins[shmemx+xshift][shmemy];
		int dH = 2 * currentSpin * ( top + right + bottom + left );

		//Retrieve and write the random value.
		int* R = ki.R;
		int randomValueIndex = globalx+globaly/2*BLOCK_SIZE;
		int randomValue = R[randomValueIndex];
		randomValue = RANDOM_A*randomValue+RANDOM_B;
		randomValue = oddrow ? (RANDOM_A*randomValue+RANDOM_B) : randomValue;
		if(oddrow)
		{
			R[randomValueIndex] = randomValue;
		}

		bool useThreshold = dH==4 or dH==8;
		float threshold = (dH==4) ? ki.exp4 : ki.exp8;
		bool meetThreshold = fabs(randomValue*4.656612e-10) < threshold;
		if ((not useThreshold) or (useThreshold and meetThreshold))
		{
			S[globalIndex]=-currentSpin;
		}
	}
}

typedef struct HeisenbergSpin
{
	float x,y,z;
} HeisenbergSpin;