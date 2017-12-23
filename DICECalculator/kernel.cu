#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h> 
#include <string>
#include <time.h>
#include <chrono>
#include <iostream>       
#include <thread>         

using namespace std;

//###############################################################################################################################
// Local Defines
//###############################################################################################################################

#define cNumberOfBlocks           (1536)
#define cNumberOfThreadsPerBlock  (256)
#define cSizeOfDataPerThread      (128)
#define cNumberOfThreads          (cNumberOfBlocks*cNumberOfThreadsPerBlock)

#ifndef STATUS
#define CUDA_E_OK                 ((uint8_t)0)
#define CUDA_NOT_OK               ((uint8_t)1)
#define CUDA_PENDING_OK           ((uint8_t)2)
#endif


#define mPRINT_TIME(func)                                                        \
	auto startTimer = chrono::steady_clock::now();                               \
	func;                                                                        \
	auto endTimer = chrono::steady_clock::now();                                 \
	cout << "Elapsed time in milliseconds : "                                    \
	<< chrono::duration_cast<chrono::milliseconds>(endTimer - startTimer).count()\
	<< " ms" << endl;

//###############################################################################################################################
// External Libs for CUDA 
//###############################################################################################################################
/*User libraries for CUDA*/
#include "DiceCalcCudaTypes.cuh"
#include "sha3Cuda.cuh"
#include "SwatchCuda.cuh"
#include "ValidationCuda.cuh"
#include "RandomGenCuda.cuh"
#include "DiceCudaCalculation.cuh"

//###############################################################################################################################
// Local Types
//###############################################################################################################################

typedef enum ProgramStates {

	//Prepare Program execution
	eProgram_Init,
	eProgram_Get_Console_Options,
	eProgram_CUDA_Allocate_Memory,
	eProgram_CUDA_Cpy_Host_Memory,
	eProgram_CUDA_CURAND_Init,

	//Loop states
	eProgram_Loop_CUDA_Fill_Random,
	eProgram_Loop_CUDA_SHA3_Random,
	eProgram_Loop_Host_Time,
	eProgram_Loop_CUDA_SHA3_DICE_Proto,
	eProgram_Loop_CUDA_Validate,
	eProgram_Loop_Host_Validate,

	//Prepare to exit
	eProgram_CUDA_Cpy_Device_Memory,
	eProgram_Host_Prepare_Check_Unit,
	eProgram_CUDA_Clean_Device_Memory,
	eProgram_Exit,

	eProgram_Count,
}EprogramStates_t;

//###############################################################################################################################
// Local Function Protorypes
//###############################################################################################################################

//void hexstr_to_char(const char* hexstr, uint8_t* bufOut, uint8_t size);
//static uint8_t* char_to_hexstr(uint8_t* pCharArrayP, uint8_t u8CountOfBytesP, uint8_t* bufOut);
static void DisplayHeader(void);

//###############################################################################################################################
// Local Data
//###############################################################################################################################

//Host-CPU
static cudaDeviceProp props;
static cudaError_t cudaStatus;
static EprogramStates_t PStates = eProgram_Init;
static bool bIsProgramRunning = true;
static uint8_t aU8Time[sizeof(uint32_t)];
auto startTimer = chrono::steady_clock::now();
auto endTimer = chrono::steady_clock::now();
static size_t sValidDiceUnitIdx = 0;
static diceUnitHex_t diceUnitValid;

//Device-GPU
static payload_t* pD_Payloads = 0;
static uint8_t* pD_U8Time = 0;
static diceProtoHEX_t* pD_Protos = 0;
static hashProtoHex_t* pD_ProtosShaHex = 0;
static bool* pD_ValidatingRes = 0;
static uint16_t* pD_U16Zeroes = 0;

//Copy on Host
static payload_t h_Payloads[cNumberOfThreads];
static diceProtoHEX_t h_Protos[cNumberOfThreads];
static hashProtoHex_t h_ProtosShaHex[cNumberOfThreads];
static bool h_ValidatingRes[cNumberOfThreads];
static uint16_t h_U16Zeroes;

//###############################################################################################################################
// Local Functions
//###############################################################################################################################

int main(int argc, char* argv[])
{
	cudaGetDeviceProperties(&props, 0);
	cudaSetDeviceFlags(cudaDeviceScheduleYield | cudaDeviceMapHost | cudaDeviceLmemResizeToMax);

	//Stub for constant input from console
	const char* addrOp = "03037a1e2905d3bf34b31f61efcb0960ef512809";
	const char* addrMin = "0204c09f6117454ab573bd166fbef7c1e4832c1f";
	const char* zeroes = "12";

	//Set default Proto
	diceProtoHEX_t dProto;
	payload_t buf_PayloadL;
	int bIsEqualL;

	//Get zeroes from string
	uint8_t aZerosL[1];
	hexstr_to_char((uint8_t*)zeroes, aZerosL, 1);

	//Show data from GPU on console
	DisplayHeader();

	while (bIsProgramRunning)
	{
		switch (PStates)
		{
		case eProgram_Init:
			cudaDeviceReset();
			cudaThreadExit();
			//Set current GPU Card (zero by Default for single GPU on system)
			cudaStatus = cudaSetDevice(0);

			//Check for Errors
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
				PStates = eProgram_CUDA_Clean_Device_Memory;
			}
			else
			{
				PStates = eProgram_Get_Console_Options;
			}
			break;

		case eProgram_Get_Console_Options:
			PStates = eProgram_CUDA_Allocate_Memory;
			break;

		case eProgram_CUDA_Allocate_Memory:
			//Allocate memory on GPU
			cudaStatus = cudaMalloc((void**)&pD_Payloads, cNumberOfThreads * sizeof(payload_t));
			cudaStatus = cudaMalloc((void**)&pD_Protos, cNumberOfThreads * sizeof(diceProtoHEX_t));
			cudaStatus = cudaMalloc((void**)&pD_U8Time, sizeof(uint32_t));
			cudaStatus = cudaMalloc((void**)&pD_ProtosShaHex, cNumberOfThreads * sizeof(hashProtoHex_t));
			cudaStatus = cudaMalloc((void**)&pD_ValidatingRes, cNumberOfThreads * sizeof(bool));
			cudaStatus = cudaMalloc((void**)&pD_U16Zeroes, sizeof(uint16_t));

			//Check for Errors
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "cudaMalloc failed on Payload!");
				PStates = eProgram_CUDA_Clean_Device_Memory;
			}
			else
			{
				PStates = eProgram_CUDA_Cpy_Host_Memory;
			}
			break;

		case eProgram_CUDA_Cpy_Host_Memory:
			//Get seed Time
			getBeats(aU8Time);

			// Copy output vector from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(pD_U8Time, aU8Time, sizeof(uint32_t), cudaMemcpyHostToDevice);
			cudaStatus = cudaMemcpy(pD_U16Zeroes, aZerosL, sizeof(uint8_t), cudaMemcpyHostToDevice);

			//Set const value
			memcpy(dProto.addrMin, addrMin, 20 * 2);
			memcpy(dProto.addrOp, addrOp, 20 * 2);
			memcpy(dProto.validZeroes, zeroes, 1 * 2);
			memset(dProto.swatchTime, 1, 4 * 2);
			memset(dProto.shaPayload, 1, 64 * 2);

			//Init data in GPU with one CPU
			for (size_t i = 0; i < cNumberOfThreads; i++)
			{
				memcpy(&h_Protos[i], &dProto, 109 * 2);
			}

			cudaStatus = cudaMemcpy(pD_Protos, &h_Protos, sizeof(h_Protos), cudaMemcpyHostToDevice);

			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "cudaMemcpy failed!");
				PStates = eProgram_CUDA_Clean_Device_Memory;
			}
			else
			{
				PStates = eProgram_CUDA_CURAND_Init;
			}
			break;

		case eProgram_CUDA_CURAND_Init:
			// Launch a kernel on the GPU with one thread for each element.
			gCUDA_CURAND_Init << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_U8Time);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				PStates = eProgram_CUDA_Clean_Device_Memory;
			}
			else
			{
				// cudaDeviceSynchronize waits for the kernel to finish, and returns
				// any errors encountered during the launch.
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess)
				{
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
					PStates = eProgram_CUDA_Clean_Device_Memory;
				}
				else
				{
					PStates = eProgram_Loop_CUDA_Fill_Random;
				}
			}
			break;

			//Loop states
		case eProgram_Loop_CUDA_Fill_Random:
			startTimer = chrono::steady_clock::now();

			// Launch a kernel on the GPU with one thread for each element.
			gCUDA_Fill_Payload << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_Payloads);
			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				PStates = eProgram_CUDA_Clean_Device_Memory;
			}
			else
			{
				// cudaDeviceSynchronize waits for the kernel to finish, and returns
				// any errors encountered during the launch.
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess)
				{
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
					PStates = eProgram_CUDA_Clean_Device_Memory;
				}
				else
				{
					// Copy output vector from GPU buffer to host memory.
					cudaStatus = cudaMemcpy(h_Payloads, pD_Payloads, cNumberOfThreads * sizeof(payload_t), cudaMemcpyDeviceToHost);
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaMemcpy failed!");
						PStates = eProgram_CUDA_Clean_Device_Memory;
					}
					PStates = eProgram_Loop_CUDA_SHA3_Random;
				}
			}
			endTimer = chrono::steady_clock::now();
			/*cout << "Fill RANDOM Elapsed time in milliseconds : "
				<< chrono::duration_cast<chrono::milliseconds>(endTimer - startTimer).count()
				<< " ms" << endl;*/
			break;

		case eProgram_Loop_CUDA_SHA3_Random:
			startTimer = chrono::steady_clock::now();

			// Launch a kernel on the GPU with one thread for each element.
			gCUDA_SHA3_Random << < cNumberOfBlocks, cNumberOfThreadsPerBlock >> > (pD_Payloads, pD_Protos);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				PStates = eProgram_CUDA_Clean_Device_Memory;
			}
			else
			{
				// cudaDeviceSynchronize waits for the kernel to finish, and returns
				// any errors encountered during the launch.
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess)
				{
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
					PStates = eProgram_CUDA_Clean_Device_Memory;
				}
				else
				{
					// Copy output vector from GPU buffer to host memory.
					cudaStatus = cudaMemcpy(h_Protos, pD_Protos, cNumberOfThreads * sizeof(diceProtoHEX_t), cudaMemcpyDeviceToHost);
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaMemcpy failed!");
						PStates = eProgram_CUDA_Clean_Device_Memory;
					}
					PStates = eProgram_Loop_Host_Time;
				}
			}
			endTimer = chrono::steady_clock::now();
			/*cout << "SHA3-512 RANDOM Elapsed time in milliseconds : "
				<< chrono::duration_cast<chrono::milliseconds>(endTimer - startTimer).count()
				<< " ms" << endl;*/
			break;

		case eProgram_Loop_Host_Time:
			//Get seed Time
			getBeats(aU8Time);

			// Copy output vector from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(pD_U8Time, aU8Time, sizeof(uint32_t), cudaMemcpyHostToDevice);

			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "cudaMemcpy failed!");
				PStates = eProgram_CUDA_Clean_Device_Memory;
			}
			else
			{
				PStates = eProgram_Loop_CUDA_SHA3_DICE_Proto;
			}

			break;

		case eProgram_Loop_CUDA_SHA3_DICE_Proto:
			startTimer = chrono::steady_clock::now();

			//Launch a kernel on the GPU with one thread for each element.
			gCUDA_SHA3_Proto << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_Protos, pD_U8Time, pD_ProtosShaHex);

			//Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				PStates = eProgram_CUDA_Clean_Device_Memory;
			}
			else
			{
				//cudaDeviceSynchronize waits for the kernel to finish, and returns
				//any errors encountered during the launch.
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess)
				{
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
					PStates = eProgram_CUDA_Clean_Device_Memory;
				}
				else
				{
					//Copy output vector from GPU buffer to host memory.
					cudaStatus = cudaMemcpy(h_Protos, pD_Protos, cNumberOfThreads * sizeof(diceProtoHEX_t), cudaMemcpyDeviceToHost);
					cudaStatus = cudaMemcpy(h_ProtosShaHex, pD_ProtosShaHex, cNumberOfThreads * sizeof(hashProtoHex_t), cudaMemcpyDeviceToHost);

					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaMemcpy failed!");
						PStates = eProgram_CUDA_Clean_Device_Memory;
					}
					PStates = eProgram_Loop_CUDA_Validate;
				}
			}
			endTimer = chrono::steady_clock::now();
			/*cout << "SHA3-512 PROTOS Elapsed time in milliseconds : "
				<< chrono::duration_cast<chrono::milliseconds>(endTimer - startTimer).count()
				<< " ms" << endl;*/
			break;

		case eProgram_Loop_CUDA_Validate:
			startTimer = chrono::steady_clock::now();

			//Launch a kernel on the GPU with one thread for each element.
			gCUDA_ValidateProtoHash << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_ProtosShaHex, pD_U16Zeroes, pD_ValidatingRes);

			//Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess)
			{
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				PStates = eProgram_CUDA_Clean_Device_Memory;
			}
			else
			{
				//cudaDeviceSynchronize waits for the kernel to finish, and returns
				//any errors encountered during the launch.
				cudaStatus = cudaDeviceSynchronize();
				if (cudaStatus != cudaSuccess)
				{
					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
					PStates = eProgram_CUDA_Clean_Device_Memory;
				}
				else
				{
					//Copy output vector from GPU buffer to host memory.
					cudaStatus = cudaMemcpy(h_ValidatingRes, pD_ValidatingRes, cNumberOfThreads * sizeof(bool), cudaMemcpyDeviceToHost);
					if (cudaStatus != cudaSuccess) {
						fprintf(stderr, "cudaMemcpy failed!");
						PStates = eProgram_CUDA_Clean_Device_Memory;
					}
					PStates = eProgram_Loop_Host_Validate;
				}
			}
			endTimer = chrono::steady_clock::now();
			/*cout << "Validating Elapsed time in milliseconds : "
				<< chrono::duration_cast<chrono::milliseconds>(endTimer - startTimer).count()
				<< " ms" << endl;*/
			break;

		case eProgram_Loop_Host_Validate:
			PStates = eProgram_Loop_CUDA_Fill_Random;
			for (size_t i = 0; i < cNumberOfThreads; i++)
			{
				if (false == h_ValidatingRes[i])
				{
					sValidDiceUnitIdx = i;
					PStates = eProgram_CUDA_Cpy_Device_Memory;
					break;
				}
			}
			break;


			//Prepare to exit
		case eProgram_CUDA_Cpy_Device_Memory:
			cudaStatus = cudaMemcpy(buf_PayloadL.payload, pD_Payloads[sValidDiceUnitIdx].payload, sizeof(payload_t), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) 
			{
				fprintf(stderr, "cudaMemcpy failed!");
				PStates = eProgram_CUDA_Clean_Device_Memory;
			}
			else
			{
				PStates = eProgram_Host_Prepare_Check_Unit;
			}
			break;

		case eProgram_Host_Prepare_Check_Unit:
			//Set Up Dice Unit
			memcpy(diceUnitValid.addrOp, dProto.addrOp, 20 * 2);
			memcpy(diceUnitValid.addrMin, dProto.addrMin, 20 * 2);
			memcpy(diceUnitValid.validZeroes, dProto.validZeroes, 1 * 2);
			dCUDA_Char_To_HexStr(aU8Time, 4, diceUnitValid.swatchTime);
			dCUDA_Char_To_HexStr(buf_PayloadL.payload, sizeof(payload_t), diceUnitValid.payload);

			
			//Check Hash of proto is as expected
			//Hash Random
			uint8_t aShaReturnL[64];
			sha3_SingleExeuction(diceUnitValid.payload, 83 * 2, aShaReturnL);

			//Save data to Global Memory in HexString
			dCUDA_Char_To_HexStr(aShaReturnL, 64, dProto.shaPayload);

			//Set Time in Global Memory for each Proto
			memcpy(dProto.swatchTime, diceUnitValid.swatchTime, 4 * 2);

			//Hash Random
			sha3_SingleExeuction(&dProto, 109 * 2, aShaReturnL);

			//Save data to Global Memory in HexString 
			uint8_t aShaHexReturnL[128];
			dCUDA_Char_To_HexStr(aShaReturnL, 64, aShaHexReturnL);

			//Check is the hash value is as expected
			bIsEqualL = memcmp(aShaHexReturnL, h_ProtosShaHex[sValidDiceUnitIdx].hashProto, 128);

			if (CUDA_E_OK == bIsEqualL)
			{
				PStates = eProgram_CUDA_Clean_Device_Memory;
			}
			else
			{
				PStates = eProgram_Loop_CUDA_Fill_Random;
			}

			break;

		case eProgram_CUDA_Clean_Device_Memory:
			cudaFree(pD_Payloads);
			cudaFree(pD_Protos);
			cudaFree(pD_U8Time);
			cudaFree(pD_ProtosShaHex);
			cudaFree(pD_ValidatingRes);
			fprintf(stderr, "Free GPU Memory\n");
			PStates = eProgram_Exit;
			break;

		case eProgram_Exit:
			uint8_t aPrintReadyL[129];
			memcpy(aPrintReadyL, h_ProtosShaHex[sValidDiceUnitIdx].hashProto, 128);
			aPrintReadyL[128] = '\0';

			printf("%s\n", aPrintReadyL);
			bIsProgramRunning = false;
			break;

		default:
			bIsProgramRunning = false;
			fprintf(stderr, "INVALID Program State !!!\n");
			break;
		}
	}


	return 0;
}

//###############################################################################################################################
// CPU - HOST - Functions
//###############################################################################################################################

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

//static void hexstr_to_char(const char* hexstr, uint8_t* bufOut, uint8_t size)
//{
//	size_t len = strlen(hexstr);
//
//	if (len % 2 == 0)
//	{
//		for (size_t i = 0, j = 0; j < size; i += 2, j++)
//		{
//			bufOut[j] = (hexstr[i] % 32 + 9) % 25 * 16 + (hexstr[i + 1] % 32 + 9) % 25;
//		}
//	}
//	else
//	{
//		//Nothing
//	}
//}
//
//static uint8_t* char_to_hexstr(uint8_t* pCharArrayP, uint8_t u8CountOfBytesP, uint8_t* bufOut)
//{
//	//Convert char array to hex string
//	for (size_t i = 0; i < u8CountOfBytesP; i++)
//	{
//		sprintf((char*)&bufOut[i * 2], "%02x", pCharArrayP[i]);
//	}
//	return (uint8_t*)bufOut;
//}


//###############################################################################################################################
// GPU - DEVICE - Functions
//###############################################################################################################################

//__global__ void  hashPayload(cudaHashPayload_t *c)
//{
//	int idx = (blockDim.x*blockIdx.x) + threadIdx.x;
//
//	sha3_SingleExeuction(c[idx].payload, 83, c[idx].shaPayload);
//}
//
//__global__ void  hashDiceUnit(cudaHashDiceUnit_t *c)
//{
//	int idx = (blockDim.x*blockIdx.x) + threadIdx.x;
//
//	sha3_SingleExeuction(c[idx].unit, 128, c[idx].shaDiceUnit);
//}
//
//__global__ void  validateUnits()
//{
//
//}

//__global__ void cudaRandPayload(payload_t *d_out)
//{
//	int i = (blockDim.x*blockIdx.x) + threadIdx.x;
//	getPayloadCuda(d_out[i].payload, cPayloadSize);
//}
