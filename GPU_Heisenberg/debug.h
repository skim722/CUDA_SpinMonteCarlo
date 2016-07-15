__device__ inline dtype spinNorm(Spin2D spin)
{
	return sqrt( spin.x*spin.x + spin.y*spin.y );
}
__device__ inline bool checkNorm(Spin2D spin)
{
	dtype tol = 1e-5;
	dtype norm = spinNorm(spin);
	if ( fabs(norm-1.0) > tol )
	{
		return false;
	}
	return true;
}
__device__ inline bool doubleCheckNorm(dtype x,dtype y)
{
	dtype norm = sqrt( x*x + y*y );
	dtype tol = 1e-5;
	if ( fabs(norm-1.0) > tol )
	{
		return false;
	}
	return true;
}
void displayCheckerboardSpins(KernelInputs& input,dtype*x0,dtype*y0,dtype*x1,dtype*y1)
{
	printf("set 0:\n");
	for(int row = 0; row <= input.maxy; ++row)
	{
		for(int col = 0; col <= input.maxx; ++col)
		{
			int index = row*input.nexty+col;
			std::cout << x0[index] << "," << y0[index] << "\t";
		}
		std::cout << std::endl;
	}
	printf("set 1:\n");
	for(int row = 0; row <= input.maxy; ++row)
	{
		for(int col = 0; col <= input.maxx; ++col)
		{
			int index = row*input.nexty+col;
			std::cout << x1[index] << "," << y1[index] << "\t";
		}
		std::cout << std::endl;
	}
}
void displayDeviceSpinComponents(KernelInputs& input)
{
	int NN = (input.L.x>>1)*input.L.y;
	dtype* x0 = new dtype[NN];
	dtype* y0 = new dtype[NN];
	dtype* x1 = new dtype[NN];
	dtype* y1 = new dtype[NN];
	cudaDeviceSynchronize();
	cudaMemcpy(x0,input.x0,NN*sizeof(dtype),cudaMemcpyDeviceToHost);
	cudaMemcpy(y0,input.y0,NN*sizeof(dtype),cudaMemcpyDeviceToHost);
	cudaMemcpy(x1,input.x1,NN*sizeof(dtype),cudaMemcpyDeviceToHost);
	cudaMemcpy(y1,input.y1,NN*sizeof(dtype),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	displayCheckerboardSpins(input,x0,y0,x1,y1);

	delete[] x0;
	delete[] y0;
	delete[] x1;
	delete[] y1;
}
bool checkSpinNorms(dtype*x0,dtype*y0,dtype*x1,dtype*y1,int n)
{
	dtype tol = 1e-5;
	for (int i=0;i<n;++i)
	{
		dtype norm0 = sqrt( x0[i]*x0[i] + y0[i]*y0[i] );
		dtype norm1 = sqrt( x1[i]*x1[i] + y1[i]*y1[i] );
		if ( norm0-1 > tol or norm1-1 > tol )
		{
			printf("Norm is not 1! ( index %d:  norm0: %f, norm1: %f )\n", i,norm0, norm1);
			return false;
		}
	}
	return true;
}
void printSpin(dtype*x,dtype*y,int index)
{
	printf("CPU index %d: %f, %f\n",index,x[index],y[index]);
}
void checkDeviceSpinNorm(KernelInputs input)
{//Transfer from GPU to sum up on CPU.
	int NN = (input.L.x>>1)*input.L.y;
	dtype* x0 = new dtype[NN];
	dtype* y0 = new dtype[NN];
	dtype* x1 = new dtype[NN];
	dtype* y1 = new dtype[NN];
	cudaDeviceSynchronize();
	cudaMemcpy(x0,input.x0,NN*sizeof(dtype),cudaMemcpyDeviceToHost);
	cudaMemcpy(y0,input.y0,NN*sizeof(dtype),cudaMemcpyDeviceToHost);
	cudaMemcpy(x1,input.x1,NN*sizeof(dtype),cudaMemcpyDeviceToHost);
	cudaMemcpy(y1,input.y1,NN*sizeof(dtype),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	displayCheckerboardSpins(input,x0,y0,x1,y1);

	delete[] x0;
	delete[] y0;
	delete[] x1;
	delete[] y1;
}