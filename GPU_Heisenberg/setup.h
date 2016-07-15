//CPU code functions and prototypes.
double std_rand();
bool MetropolisStep (double delta_max, double T);
int getN();
void getKernelInputs(
	int* Lx,int* Ly,int* Lz,
	dtype*Hx,dtype*Hy,dtype*Hz,
	dtype*J,dtype*isingAxisX,dtype*isingAxisY,
	dtype*T);

void fillKernelArrays(KernelInputs& input)
{
	//First lets check to see if the sizes are as expected.
	if((input.L.x & 1) or (input.L.y & 1))
	{
		printf("ERROR! Domain sizes must be even-numbered! Exiting.\n");
		exit(EXIT_FAILURE);
	}
	if((input.L.x>>1) % THREADS_PER_BLOCK != 0)
	{
		printf("ERROR! Threads per block (%d) must evenly divide into the compressed checkerboard dimension! (Lx / 2 = %d)\n",
			THREADS_PER_BLOCK,input.L.x>>1);
		exit(EXIT_FAILURE);
	}
	if(input.L.y % LINE_LENGTH != 0)
	{
		printf("ERROR! Vertical line length must evenly divide into the spin dimension! (Ly)\n");
		exit(EXIT_FAILURE);
	}
	//Now allocate
	int checkerboardSize = (input.L.x>>1)*input.L.y;
	cudaMalloc((void**) &input.x0, checkerboardSize*sizeof(dtype));
	cudaMalloc((void**) &input.y0, checkerboardSize*sizeof(dtype));
	cudaMalloc((void**) &input.x1, checkerboardSize*sizeof(dtype));
	cudaMalloc((void**) &input.y1, checkerboardSize*sizeof(dtype));
	int randomNumberSize = (input.L.x>>1)*(input.L.y/LINE_LENGTH);
	cudaMalloc((void**) &input.r, randomNumberSize*sizeof(unsigned int));
	//Compute derived variables.
	input.nexty = input.L.x >> 1;
	input.maxx = input.nexty - 1;
	input.maxy = input.L.y - 1;
}

void initializeInputArrays(KernelInputs& input)
{
	// initialize spins: all pointing along +/- the same direction, isingAxis
	// printf("Allocating host arrays.\n");
	int checkerboardSize = (input.L.x>>1)*input.L.y;
	dtype* x0 = new dtype[checkerboardSize];
	dtype* y0 = new dtype[checkerboardSize];
	dtype* x1 = new dtype[checkerboardSize];
	dtype* y1 = new dtype[checkerboardSize];
	dtype iix = input.isingAxis.x;
	dtype iiy = input.isingAxis.y;
	//printf("iix,iiy: %f,%f\n",iix,iiy);
	// printf("Initializing spins.\n");
	// for (int i = 0; i < input.L.x; ++i)
	// {
	// 	for (int j = 0; j < input.L.y; ++j)
	// 	{
	// 		for (int k = 0; k < 1; ++k)
	// 		{
	// 			dtype s = std_rand() < 0.5 ? +1. : -1.;   // random flip
	// 			int index = i>>1 + j*input.nexty;
	// 			bool oddrow = j & 1;
	// 			bool oddcol = i & 1;
	// 			if( oddrow )
	// 			{
	// 				if( oddcol )
	// 				{
	// 					x0[index] = s*iix;
	// 					y0[index] = s*iiy;
	// 				}
	// 				else
	// 				{
	// 					x1[index] = s*iix;
	// 					y1[index] = s*iiy;
	// 				}
	// 			}
	// 			else
	// 			{
	// 				if( oddcol )
	// 				{
	// 					x1[index] = s*iix;
	// 					y1[index] = s*iiy;
	// 				}
	// 				else
	// 				{
	// 					x0[index] = s*iix;
	// 					y0[index] = s*iiy;
	// 				}
	// 			}
	// 		}
	// 	}
	// }
	for (int row = 0; row <= input.maxy; ++row)
	{
		for (int col = 0; col <= input.maxx; ++col)
		{
			for (int k = 0; k < 1; ++k)
			{
				int index = col + row*input.nexty;
				dtype s = std_rand() < 0.5 ? +1. : -1.;   // random flip
				x0[index] = s*iix;
				y0[index] = s*iiy;
				s = std_rand() < 0.5 ? +1. : -1.;   // random flip
				x1[index] = s*iix;
				y1[index] = s*iiy;
			}
		}
	}
	// if ( not checkSpinNorms(x0,y0,x1,y1,checkerboardSize) )
	// 	exit(EXIT_FAILURE);
	// else
	// 	printf("Spins after initialization are of norm 1.\n");

	// printf("Initialized spins.\n");
	//Initialize random numbers
	int randomNumberSize = (input.L.x>>1)*(input.L.y/LINE_LENGTH);
	unsigned int* r = new unsigned int[checkerboardSize];
	for(int i=0;i<(input.L.x>>1);++i)
	{
		for(int j=0;j<(input.L.y/LINE_LENGTH);++j)
		{
			unsigned int index = i + j*input.nexty;
			r[index] = index;
		}
	}
	//Transfer to device.
	cudaMemcpy(input.x0,x0,checkerboardSize*sizeof(dtype),cudaMemcpyHostToDevice);
	cudaMemcpy(input.y0,y0,checkerboardSize*sizeof(dtype),cudaMemcpyHostToDevice);
	cudaMemcpy(input.x1,x1,checkerboardSize*sizeof(dtype),cudaMemcpyHostToDevice);
	cudaMemcpy(input.y1,y1,checkerboardSize*sizeof(dtype),cudaMemcpyHostToDevice); 
	cudaMemcpy(input.r,r,randomNumberSize*sizeof(unsigned int),cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	// checkDeviceSpinNorm(input);
	// printf("Spins on device are of norm 1 after transfer from host.\n");
	//Delete host arrays.
	delete[] x0;
	delete[] y0;
	delete[] x1;
	delete[] y1;
	delete[] r;
}

void destroyKernelInputArrays(KernelInputs& input)
{
	cudaFree(input.x0);
	cudaFree(input.y0);
	cudaFree(input.x1);
	cudaFree(input.y1);
	cudaFree(input.r);
}