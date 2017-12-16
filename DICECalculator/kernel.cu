#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h> 
#include <time.h>
#include <chrono>

/*User libraries for CUDA*/
#include "sha3Cuda.cuh"
#include "SwatchCuda.cuh"
#include "RandomGenCuda.cuh"
#include "ValidationCuda.cuh"
#include "DiceCalcCuda.cuh"

//Local defines
#define cArraySize   (1024*1024)
#define cAddressSize (20)

typedef struct hashing
{
	bool isValid;
	uint8_t buffer[SHA3_512];
}hasinig_t;

using namespace std;

//Local Function prototypes
cudaError_t hashWithCuda(hasinig_t *c, unsigned int size);
cudaError_t randWithCuda(payload_t *c, unsigned int size);

void hexstr_to_char(const char* hexstr, uint8_t* bufOut, uint8_t size);

static void DisplayHeader();
//static void calcSHA3(void);

//Local data
cudaDeviceProp props;
static hasinig_t c[cArraySize];
static payload_t diceUnits[cArraySize];

//Function Executed on GPU
__global__ void  calculateDICE(hasinig_t *c)
{
	int i = (blockDim.x*blockIdx.x) + threadIdx.x;

	sha3_SingleExeuction("Hello World", 12, c[i].buffer);

	c[i].isValid = true;
}

__global__ void cudaRandPayload(payload_t *d_out)
{
	int i = (blockDim.x*blockIdx.x) + threadIdx.x;
	getPayloadCuda(d_out[i].payload, cPayloadSize);
}

int main(int argc, char* argv[])
{
	cudaGetDeviceProperties(&props, 0);
	cudaSetDeviceFlags(cudaDeviceScheduleYield | cudaDeviceMapHost | cudaDeviceLmemResizeToMax);
	const int size = cArraySize;

	//In HEX
	const char* addrOp = "03037a1e2905d3bf34b31f61efcb0960ef512809";
	const char* addrMin = "0204c09f6117454ab573bd166fbef7c1e4832c1f";
	const char* zeroes = "13";

	uint8_t Array[cAddressSize];

	hexstr_to_char(addrOp, Array, cAddressSize);
	hexstr_to_char(addrMin, Array, cAddressSize);
	hexstr_to_char(zeroes, Array, 1);

	DisplayHeader();

	memset(c, 0, size);

	auto startTimer = chrono::steady_clock::now();
    cudaError_t cudaStatus = hashWithCuda(c, size);
	auto endTimer = chrono::steady_clock::now();

	wcout << endl << c << endl;

	cout << "Threads per second : "
		<< (cArraySize/chrono::duration_cast<chrono::milliseconds>(endTimer - startTimer).count())*1000
		<<  endl;

	cout << "Elapsed time in milliseconds : "
		<< chrono::duration_cast<chrono::milliseconds>(endTimer - startTimer).count()
		<< " ms" << endl;

	startTimer = chrono::steady_clock::now();
	////for (size_t i = 0; i < props.maxThreadsDim[0]* props.maxThreadsDim[1]; i++)
	////{
	////	uint8_t bufferL[SHA3_512];
	////	sha3_SingleExeuction("Hello World", 12, bufferL);
	////}
	//for (int i = 0; i < cArraySize; i++)
	//{
	//	getPayload(diceUnits[i].payload, cPayloadSize);
	//}
	cudaStatus = randWithCuda(diceUnits, size);

	endTimer = chrono::steady_clock::now();

	cout << "Elapsed time in nanoseconds : "
		<< chrono::duration_cast<chrono::milliseconds>(endTimer - startTimer).count()
		<< " ms" << endl;

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t hashWithCuda(hasinig_t *c, unsigned int size)
{
	hasinig_t *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(hasinig_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    calculateDICE <<< props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2] >>>(dev_c);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(hasinig_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
Error:
    cudaFree(dev_c);
    
    return cudaStatus;
}

cudaError_t randWithCuda(payload_t *c, unsigned int size)
{
	payload_t *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(payload_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	cudaRandPayload <<< props.maxThreadsDim[0]*6, props.maxThreadsDim[1]/6, props.maxThreadsDim[2] >>>(dev_c);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(payload_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
Error:
	cudaFree(dev_c);

	return cudaStatus;
}

static void DisplayHeader()
{
	const int kb = 1024;
	const int mb = kb * kb;
	wcout << "NBody.GPU" << endl << "=========" << endl << endl;

	wcout << "CUDA version:   v" << CUDART_VERSION << endl;
	//wcout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl;

	int devCount;
	cudaGetDeviceCount(&devCount);
	wcout << "CUDA Devices: " << endl << endl;


	for (int i = 0; i < devCount; ++i)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		wcout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
		wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
		wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
		wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
		wcout << "  Block registers: " << props.regsPerBlock << endl << endl;

		wcout << "  Warp size:         " << props.warpSize << endl;
		wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
		wcout << "  Multiprocessors: " << props.multiProcessorCount << endl;
		wcout << "  Threads per multiprocessor: " << props.maxThreadsPerMultiProcessor << endl;
		wcout << "  Concurent kernels: " << props.concurrentKernels << endl;
		wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << endl;
		wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << endl;
		wcout << endl;
	}

}

void hexstr_to_char(const char* hexstr, uint8_t* bufOut, uint8_t size)
{
	size_t len = strlen(hexstr);

	if (len % 2 == 0)
	{
		for (size_t i = 0, j = 0; j < size; i += 2, j++)
		{
			bufOut[j] = (hexstr[i] % 32 + 9) % 25 * 16 + (hexstr[i + 1] % 32 + 9) % 25;
		}
	}
	else
	{
		//Nothing
	}
}