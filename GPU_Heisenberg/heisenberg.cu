#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include "StopWatch.h"
#include "common.h"

#include "debug.h"
#include "device.h"
#include "setup.h"

__global__ void oneMonteCarloStepPerSpin2D(KernelInputs inputs,bool set0,dtype ratioFactor)
{
	//Indexing variables.
	int globalx = blockIdx.x*blockDim.x + threadIdx.x;
	int globalyStart = blockIdx.y*LINE_LENGTH;
	// int nexty = inputs.L.x >> 1;
	// int maxx = nexty - 1;
	// int maxy = inputs.L.y - 1;
	int currentIndex = globalx + globalyStart*inputs.nexty;//start index.
	//Get starting random number.
	unsigned int randomInt = inputs.r[globalx + blockIdx.y*inputs.nexty];
	//Get correct set.
	dtype* x = (set0) ? inputs.x0 : inputs.x1;
	dtype* y = (set0) ? inputs.y0 : inputs.y1;
	dtype* otherx = (set0) ? inputs.x1 : inputs.x0;
	dtype* othery = (set0) ? inputs.y1 : inputs.y0;

	// dtype mag = sqrt(x[currentIndex]*x[currentIndex]+y[currentIndex]*y[currentIndex]);
	// if(mag-1.0 > 1e-5) printf("Start vector is not unit!!! %f\n",mag);

	//First stencil.
	Spin2D trial_spin,try_spin,top,main,off,bottom;//Stencil spins. 'main' and 'off' are left and right, depending on the row.
	Spin2D delta;
	//Spin2D neighbourSum;
	dtype randomValue1,randomValue2;
	dtype e0,e1;
	double ratio;
	//int delta_ss,mup;

	//Get trial spin.
	trial_spin.x = x[currentIndex];
	trial_spin.y = y[currentIndex];
	if(not checkNorm(trial_spin))
	{
		printf("read spin does not have norm of 1. (set0 %d index %d (%d,%d): %f: %f, %f)\n",set0,currentIndex,globalx,globalyStart,spinNorm(trial_spin),trial_spin.x,trial_spin.y);
		printf("blockidx,y: %d,%d; threadidx,y: %d,%d; blockDimx,y: %d,%d\n",
			blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,blockDim.x,blockDim.y);
	}
	if(not doubleCheckNorm(x[currentIndex],y[currentIndex]))
	{
		printf("(double check) read spin does not have norm of 1. (index %d: %f: %f, %f)\n",
			currentIndex,sqrt( x[currentIndex]*x[currentIndex] + y[currentIndex]*y[currentIndex] ),
			x[currentIndex],y[currentIndex]);
	}
	//Get neighbors.
	top.x = otherx[ (blockIdx.y==0) ? (currentIndex + inputs.maxy*inputs.nexty) : (currentIndex - inputs.nexty) ];
	top.y = othery[ (blockIdx.y==0) ? (currentIndex + inputs.maxy*inputs.nexty) : (currentIndex - inputs.nexty) ];
	main.x = otherx[currentIndex];
	main.y = othery[currentIndex];
	off.x = otherx[currentIndex + determineOffset(0,set0,globalx,inputs.maxx)];
	off.y = othery[currentIndex + determineOffset(0,set0,globalx,inputs.maxx)];
	bottom.x = otherx[currentIndex+inputs.nexty];
	bottom.y = othery[currentIndex+inputs.nexty];
	//Calculate current energy.
	e0 = dEnergydSpin2D(trial_spin,top,main,off,bottom,inputs.J,inputs.Hx,inputs.Hy);
	//Generate random values.
	randomInt = RANDOM_A*randomInt + RANDOM_B;
	randomValue1 = randomInt*inputs.normRandom;
	randomInt = RANDOM_A*randomInt + RANDOM_B;
	randomValue2 = randomInt*inputs.normRandom;
	//Generate perturbed spin.
	delta = makeDelta(inputs.delta_max,randomValue1,randomValue2);
	try_spin.x = trial_spin.x;try_spin.y = trial_spin.y;
	try_move2D(try_spin,delta);
	if(not checkNorm(try_spin))	printf("try_spin does not have norm of 1 before write.\n");
	//Calculate new energy.
	e1 = dEnergydSpin2D(try_spin,top,main,off,bottom,inputs.J,inputs.Hx,inputs.Hy);
	ratio = exp(-(e1-e0)*ratioFactor);//exp(-(e1-e0)*ratioFactor);
	//Sum neighbours and compute thresholds.
	// neighbourSum.x = top.x + main.x + off.x + bottom.x;
	// neighbourSum.y = top.y + main.y + off.y + bottom.y;
	// delta_ss = rint( 2.0 * (try_spin.x * neighbourSum.x + try_spin.y * neighbourSum.y ) );
	// mup = rint( try_spin.x * inputs.isingAxis.x + try_spin.y * inputs.isingAxis.y );
	//Compare.
	randomInt = RANDOM_A*randomInt + RANDOM_B;
	if ( randomInt*inputs.normRandom < ratio )
	{
		// dtype mag = sqrt(try_spin.x*try_spin.x+try_spin.y*try_spin.y);
		// if(mag-1.0 > 1e-5) printf("try_move does not give unit vector!!!!! %f\n",mag);
		if(not checkNorm(try_spin))
			printf("try_spin does not have norm of 1 before write.\n");
		x[currentIndex] = try_spin.x;
		y[currentIndex] = try_spin.y;
	}

	for(int i=1;i<LINE_LENGTH-1;++i)
	{
		//Increment currentIndex.
		currentIndex += inputs.nexty;
		//Reuse last data.
		top = main;
		main = bottom;
		//Get trial spin.
		trial_spin.x = x[currentIndex];
		trial_spin.y = y[currentIndex];
		if(not checkNorm(trial_spin))
		{
			printf("read spin does not have norm of 1. (set0 %d index %d (%d,%d): %f: %f, %f)\n",set0,currentIndex,globalx,globalyStart,spinNorm(trial_spin),trial_spin.x,trial_spin.y);
		}
		if(not doubleCheckNorm(x[currentIndex],y[currentIndex]))
		{
			printf("(double check) read spin does not have norm of 1. (index %d: %f: %f, %f)\n",
				currentIndex,sqrt( x[currentIndex]*x[currentIndex] + y[currentIndex]*y[currentIndex] ),
				x[currentIndex],y[currentIndex]);
			printf("blockidx,y: %d,%d; threadidx,y: %d,%d; blockDimx,y: %d,%d\n",
				blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,blockDim.x,blockDim.y);
		}
		//Get neighbors.
		off.x = otherx[currentIndex + determineOffset(i,set0,globalx,inputs.maxx)];
		off.y = othery[currentIndex + determineOffset(i,set0,globalx,inputs.maxx)];
		bottom.x = otherx[currentIndex+inputs.nexty];
		bottom.y = othery[currentIndex+inputs.nexty];
		//Calculate current energy.
		e0 = dEnergydSpin2D(trial_spin,top,main,off,bottom,inputs.J,inputs.Hx,inputs.Hy);
		//Generate random values.
		randomInt = RANDOM_A*randomInt + RANDOM_B;
		randomValue1 = randomInt*inputs.normRandom;
		randomInt = RANDOM_A*randomInt + RANDOM_B;
		randomValue2 = randomInt*inputs.normRandom;
		//Generate perturbed spin.
		delta = makeDelta(inputs.delta_max,randomValue1,randomValue2);
		try_spin.x = trial_spin.x;try_spin.y = trial_spin.y;
		try_move2D(try_spin,delta);
		//Calculate new energy.
		e1 = dEnergydSpin2D(try_spin,top,main,off,bottom,inputs.J,inputs.Hx,inputs.Hy);
		ratio = exp(-(e1-e0)*ratioFactor);//exp(-(e1-e0)/(T+1e-8));
		//Sum neighbours and compute thresholds.
		//neighbourSum.x = top.x + main.x + off.x + bottom.x;
		//neighbourSum.y = top.y + main.y + off.y + bottom.y;
		//delta_ss = rint( 2.0 * (try_spin.x * neighbourSum.x + try_spin.y * neighbourSum.y ) );
		//mup = rint( try_spin.x * inputs.isingAxis.x + try_spin.y * inputs.isingAxis.y );
		//Compare.
		randomInt = RANDOM_A*randomInt + RANDOM_B;
		if ( randomInt*inputs.normRandom < ratio )
		{
			// dtype mag = sqrt(try_spin.x*try_spin.x+try_spin.y*try_spin.y);
			// if(mag-1.0 > 1e-5) printf("try_move does not give unit vector!!!!! %f\n",mag);
			if(not checkNorm(try_spin))
				printf("try_spin does not have norm of 1 before write.\n");
			x[currentIndex] = try_spin.x;
			y[currentIndex] = try_spin.y;
		}
	}
	//Last stencil.
	//Increment currentIndex.
	currentIndex += inputs.nexty;
	//Reuse last data.
	top = main;
	main = bottom;
	//Get trial spin.
	trial_spin.x = x[currentIndex];
	trial_spin.y = y[currentIndex];
	if(not checkNorm(trial_spin))
	{
		printf("read spin does not have norm of 1. (set0 %d index %d (%d,%d): %f: %f, %f)\n",set0,currentIndex,globalx,globalyStart,spinNorm(trial_spin),trial_spin.x,trial_spin.y);
	}
	if(not doubleCheckNorm(x[currentIndex],y[currentIndex]))
	{
		printf("(double check) read spin does not have norm of 1. (index %d: %f: %f, %f)\n",
			currentIndex,sqrt( x[currentIndex]*x[currentIndex] + y[currentIndex]*y[currentIndex] ),
			x[currentIndex],y[currentIndex]);
		printf("blockidx,y: %d,%d; threadidx,y: %d,%d; blockDimx,y: %d,%d\n",
			blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,blockDim.x,blockDim.y);
	}
	//Get neighbors.
	off.x = otherx[currentIndex + determineOffset((LINE_LENGTH-1),set0,globalx,inputs.maxx)];
	off.y = othery[currentIndex + determineOffset((LINE_LENGTH-1),set0,globalx,inputs.maxx)];
	bottom.x = otherx[ (blockIdx.y==(gridDim.y-1)) ? globalx : (currentIndex+inputs.nexty) ];
	bottom.y = othery[ (blockIdx.y==(gridDim.y-1)) ? globalx : (currentIndex+inputs.nexty) ];
	//Calculate current energy.
	e0 = dEnergydSpin2D(trial_spin,top,main,off,bottom,inputs.J,inputs.Hx,inputs.Hy);
	//Generate random values.
	randomInt = RANDOM_A*randomInt + RANDOM_B;
	randomValue1 = randomInt*inputs.normRandom;
	randomInt = RANDOM_A*randomInt + RANDOM_B;
	randomValue2 = randomInt*inputs.normRandom;
	//Generate perturbed spin.
	delta = makeDelta(inputs.delta_max,randomValue1,randomValue2);
	try_spin.x = trial_spin.x;try_spin.y = trial_spin.y;
	try_move2D(try_spin,delta);
	//Calculate new energy.
	e1 = dEnergydSpin2D(try_spin,top,main,off,bottom,inputs.J,inputs.Hx,inputs.Hy);
	ratio = exp(-(e1-e0)*ratioFactor);//exp(-(e1-e0)/(T+1e-8));
	//Sum neighbours and compute thresholds.
	//neighbourSum.x = top.x + main.x + off.x + bottom.x;
	//neighbourSum.y = top.y + main.y + off.y + bottom.y;
	//delta_ss = rint( 2.0 * (try_spin.x * neighbourSum.x + try_spin.y * neighbourSum.y ) );
	//mup = rint( try_spin.x * inputs.isingAxis.x + try_spin.y * inputs.isingAxis.y );
	//Compare.
	randomInt = RANDOM_A*randomInt + RANDOM_B;
	if ( randomInt*inputs.normRandom < ratio )
	{
		// dtype mag = sqrt(try_spin.x*try_spin.x+try_spin.y*try_spin.y);
		// if(mag-1.0 > 1e-5) printf("try_move does not give unit vector!!!!! %f\n",mag);
		if(not checkNorm(try_spin))
			printf("try_spin does not have norm of 1 before write.\n");
		x[currentIndex] = try_spin.x;
		y[currentIndex] = try_spin.y;
	}

	//Write out random number.
	inputs.r[globalx + blockIdx.y*inputs.nexty] = randomInt;

	// mag = sqrt(x[currentIndex]*x[currentIndex]+y[currentIndex]*y[currentIndex]);
	// if(mag-1.0 > 1e-5) printf("Vector at end is not unit!!! %f\n",mag);
}

void stepTiming()
{
	printf("Starting Timing Tests.\n");

	//Setup
	double delta_max = 0.5;
	double T = 200;
	KernelInputs input;
	getKernelInputs(
		&input.L.x,&input.L.y,&input.L.z,
		&input.Hx,&input.Hy,&input.Hz,
		&input.J,&input.isingAxis.x,&input.isingAxis.y,
		&input.T);
	input.normRandom = NORMALIZE_RANDOM;
	fillKernelArrays(input);
	initializeInputArrays(input);
	int N = input.L.x*input.L.y;
	KernelDecomposition kd;
	kd.blocksPerGrid.x = input.L.x / THREADS_PER_BLOCK;
	kd.blocksPerGrid.y = input.L.y / LINE_LENGTH;
	kd.threadsPerBlock.x = THREADS_PER_BLOCK;
	kd.threadsPerBlock.y = 1;
	dim3 grid(kd.blocksPerGrid.x,kd.blocksPerGrid.y);
	dim3 block(kd.threadsPerBlock.x,kd.threadsPerBlock.y);

	//Timer
	StopWatch stopwatch;
	
	//CPU
	stopwatch.start();
	for(int i=0;i<N;++i)
	{
		MetropolisStep(delta_max,T);
	}
	float cputime = stopwatch.stop();

	printf("CPU Time: %f ms\n",cputime);

	//GPU
	stopwatch.start();
	oneMonteCarloStepPerSpin2D<<<grid,block>>>(input,true,T);
	oneMonteCarloStepPerSpin2D<<<grid,block>>>(input,false,T);
	cudaDeviceSynchronize();
	float gputime = stopwatch.stop();

	printf("GPU Time: %f ms\n",gputime);

	printf("Speedup: %f\n",cputime/gputime);

	destroyKernelInputArrays(input);
}

typedef struct ReducedQuantities
{
	dtype magnetization;
	dtype magnetizationPerSpin;
	dtype energyPerSpin;
} ReducedQuantities;

ReducedQuantities serialReduce2D(KernelInputs input)
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

	// if(not checkSpinNorms(x0,y0,x1,y1,NN))
	// 	exit(EXIT_FAILURE);

	ReducedQuantities rq;
	dtype energy1 = 0;
	dtype mx = 0;
	dtype my = 0;
	for(int col=0;col<=input.maxx;++col)
	{
		for(int row=0;row<=input.maxy;++row)
		{
			int index = col + row*input.nexty;
			mx += x0[index] + x1[index];
			my += y0[index] + y1[index];
			int forwardy = (row == 0) ? (input.maxy*input.nexty) : (-input.nexty);
			bool oddrow = row & 1;
			int forwardx0 = oddrow ? 1 : 0;
			int forwardx1 = oddrow ? 0 : 1;
			energy1 += x0[index]*x1[index+forwardx1]+y0[index]*y1[index+forwardy];
			energy1 += x1[index]*x0[index+forwardx0]+y1[index]*y0[index+forwardy];
		}
	}
	int N = NN << 1;
	rq.magnetization = sqrt( mx*mx + my*my );
	rq.magnetizationPerSpin = rq.magnetization / N;
	rq.energyPerSpin = -( input.J*energy1 + input.Hx*mx + input.Hy*my ) / N;

	delete[] x0;
	delete[] y0;
	delete[] x1;
	delete[] y1;

	return rq;
}

void gpuSimulation(int MonteCarloSteps)
{
	//Setup parameters.
	KernelInputs input;
	input.delta_max = 0.5;
	getKernelInputs(
		&input.L.x,&input.L.y,&input.L.z,
		&input.Hx,&input.Hy,&input.Hz,
		&input.J,&input.isingAxis.x,&input.isingAxis.y,
		&input.T);
	input.normRandom = NORMALIZE_RANDOM;
	dtype T = input.T;
	fillKernelArrays(input);
	initializeInputArrays(input);
	KernelDecomposition kd;
	kd.blocksPerGrid.x = (input.L.x >> 1) / THREADS_PER_BLOCK;
	kd.blocksPerGrid.y = input.L.y / LINE_LENGTH;
	kd.threadsPerBlock.x = THREADS_PER_BLOCK;
	kd.threadsPerBlock.y = 1;
	dim3 grid(kd.blocksPerGrid.x,kd.blocksPerGrid.y);
	dim3 block(kd.threadsPerBlock.x,kd.threadsPerBlock.y);

	//Output file setup.
	std::string quenchingDataFileName ("heisenberg_quenching_gpu.data");
	std::ofstream quenchingDataFile(quenchingDataFileName.c_str());
   	quenchingDataFile 	<< " Lx,Ly (Domain Size): "<<input.L.x<<","<<input.L.y<<","<<'\t'
   						<< " Hx,Hy (Magnetic Field): "<<input.Hx<<","<<input.Hy<<","<<'\t'
   						<<" Monte Carlo Steps: "<< MonteCarloSteps << std::endl;

	//checkDeviceSpinNorm(input);

   	//Start Simulation!
	StopWatch stopwatch;
	stopwatch.start();
	printf("Starting GPU Simulation.\n");
	for (int t=0;t<200;t++)
	{
		dtype ratioFactor = 1.0/(T+1e-8);
		int thermSteps = int(0.2 * MonteCarloSteps);
		double mAv = 0, m2Av = 0, eAv = 0, e2Av = 0, MAv=0;
		// checkDeviceSpinNorm(input);
		// printf("The spin norms are still 1 just before therm steps.\n");
		//displayDeviceSpinComponents(input);
		for (int i = 0; i < thermSteps; i++)
		{
			oneMonteCarloStepPerSpin2D<<<grid,block>>>(input,true,ratioFactor);
			oneMonteCarloStepPerSpin2D<<<grid,block>>>(input,false,ratioFactor);
		}
		// checkDeviceSpinNorm(input);
		// printf("The spin norms are still 1 after therm steps.\n");
		// exit(EXIT_SUCCESS);
		for (int mc_n = 0; mc_n < MonteCarloSteps; mc_n++)
		{
			//checkDeviceSpinNorm(input);
			oneMonteCarloStepPerSpin2D<<<grid,block>>>(input,true,ratioFactor);
			oneMonteCarloStepPerSpin2D<<<grid,block>>>(input,false,ratioFactor);
			
			//printf("MC step: %d\n",mc_n);
			//checkDeviceSpinNorm(input);
			// double m = fabs(magnetizationPerSpin());
			// double M = fabs(magnetization());
			// double e = energyPerSpin();
			ReducedQuantities rq = serialReduce2D(input);
			double m = fabs(rq.magnetizationPerSpin);
			double M = fabs(rq.magnetization);
			double e = rq.energyPerSpin;
			
			MAv +=M;
			mAv += m; m2Av += m * m;
			eAv += e; e2Av += e * e;
		}

		mAv /= MonteCarloSteps; m2Av /= MonteCarloSteps;
		eAv /= MonteCarloSteps; e2Av /= MonteCarloSteps;
		MAv /= MonteCarloSteps;

		//Final print out.
		/* Convert to work with GPU data structures.
		std::cout<<"Final Spin Set at T="<<T<<'\n'<<std::endl;
		for (int i = 0; i < input.L.x; i++)
		{
			for (int j = 0; j < input.L.y; j++)
			{
				for (int k = 0; k < input.L.z; k++)
				{
					for (int l=0;l<DIMENSIONS;l++)
					{
						cout<<spins[i][j][k][l]<<'\t';
					}
					std::cout << "  ";
				}
				std::cout << '\n' << endl;;
			}
			std::cout<<'\n\n'<<endl;
		}
		*/
		std::cout <<"iteration number : "<<t<<"\n"<<std::endl;
		std::cout <<"T ="<<'\t'<<T<< " <m> = "<<'\t' << mAv << " +/- " << sqrt(fabs(m2Av - mAv*mAv)) <<'\t'<<  " <e> = " <<'\t'<< eAv << " +/- " << sqrt(e2Av - eAv*eAv) << std::endl;
		quenchingDataFile <<"T"<<'\t'<<  T <<'\t' << " <m> " <<'\t'<< mAv <<"   "<<'\t'<<"M"<<'\t'<<MAv<< std::endl;
		T=T-(T/1000)*(t);
	}
	//Stop timer and print execution time.
	cudaDeviceSynchronize();
	float gputime = stopwatch.stop();
	printf("GPU Time: %f ms\n",gputime);
	//Notify and close file.
	std::cout 	<< " \n Magnetization and energy per spin written in file "
			<< "\""<<quenchingDataFileName<<"\"" << std::endl;
	quenchingDataFile.close();
	//Cleanup.
	destroyKernelInputArrays(input);
}