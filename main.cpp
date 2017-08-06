#include "utils.h"

#include "gpuTimer.h"
#include "scanArray.h"

#include <iostream>
#include <fstream>

#if !defined(WIN32)
#include <sys/time.h> // for clock_gettime()

#else
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 

// MSVC defines this in winsock2.h!?
/*typedef struct timeval {
	long tv_sec;
	long tv_usec;
} timeval;
*/
int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
	// Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
	// This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
	// until 00:00:00 January 1, 1970 
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}
#endif

typedef int MY_TYPE;

int main(int argc, char **argv)
{
    
    const size_t num_elements =  (1 << 27) - 51;

	unsigned int mem_size = sizeof(MY_TYPE) * num_elements;

    MY_TYPE* h_data = (MY_TYPE*) malloc(mem_size);
      
    // initialize the input data on the host
    for( size_t i = 0; i < num_elements; ++i)
    {
        h_data[i] = 1;
    }

    MY_TYPE h_lastSum;
    MY_TYPE *d_idata, *d_odata, *d_lastSum;
    checkCudaErrors( cudaMalloc( (void**) &d_idata, mem_size));
    checkCudaErrors( cudaMalloc( (void**) &d_odata, mem_size));
	checkCudaErrors(cudaMalloc((void**)&d_lastSum, sizeof(MY_TYPE)));
    
    // copy host memory to device input array
    checkCudaErrors( cudaMemcpy( d_idata, h_data, mem_size, cudaMemcpyHostToDevice) );
    
	GpuTimer timer;
	timer.Start();

    exclusive_scan(d_odata, d_idata, num_elements, d_lastSum);
    
	timer.Stop();

	cudaDeviceSynchronize(); 
    checkCudaErrors(cudaGetLastError());

	printf("Running parallel prefix sum (prescan) of %lu elements\n", num_elements);
	printf("CUDA ran in: %f msecs.\n", timer.Elapsed());

    h_data = (MY_TYPE*) malloc(mem_size);
	checkCudaErrors(cudaMemcpy(h_data, d_odata, mem_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&h_lastSum, d_lastSum, sizeof(int), cudaMemcpyDeviceToHost));

	printf("h_lastSum=%zu, expected=%lu\n", h_lastSum, num_elements);
    
#if 1
	std::ofstream myfile("result_data.txt");
	if (myfile.is_open())
	{
		for (size_t i = 0; i < num_elements; i++)
		{
			if ((size_t)(h_data[i]) != i)
			{
				printf("FIRST ERR:i=%zu, h_data[i]=%zu\n", i, h_data[i]);
                //myfile << std::setw(12) << i << std::fixed <<
                //std::setprecision(2) << std::setw(20) << h_data[i] << std::endl;
                break;
			}
			/*myfile << std::setw(12) << i << std::fixed <<
				std::setprecision(2) << std::setw(20) << h_data[i] << std::endl;*/
		}
		myfile.close();
	}
#endif
    struct timeval start, end;
    long secs_used,micros_used;
    
    gettimeofday(&start, NULL);
    for (unsigned int i = 1; i < num_elements; i++)
    {
        h_data[i] += h_data[i-1];
    }
    gettimeofday(&end, NULL);
    
    secs_used=(end.tv_sec - start.tv_sec); //avoid overflow by subtracting first
    micros_used= ((secs_used*1000000) + end.tv_usec) - (start.tv_usec);
    printf("CPU micros_used: %.3fms\n",micros_used/1000.);
    
    free( h_data);
    //free( reference);
    cudaFree( d_odata);
    cudaFree( d_idata);

#if defined(WIN32)
	getchar();
#endif

	return 0;
}

