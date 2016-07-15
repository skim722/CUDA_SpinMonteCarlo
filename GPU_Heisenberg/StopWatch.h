#ifndef __CUDA_STOPWATCH__

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

#endif