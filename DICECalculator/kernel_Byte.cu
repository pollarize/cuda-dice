///*
//* Copyright (c) 2017, Mihail Maldzhanski <pollarize@gmail.com>
//* All rights reserved.
//*
//* Redistribution and use in source and binary forms, with or without
//* modification, are permitted provided that the following conditions are met:
//*
//* * Redistributions of source code must retain the above copyright notice, this
//*   list of conditions and the following disclaimer.
//* * Redistributions in binary form must reproduce the above copyright notice,
//*   this list of conditions and the following disclaimer in the documentation
//*   and/or other materials provided with the distribution.
//*
//* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//* POSSIBILITY OF SUCH DAMAGE.
//*/
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <iostream>
//#include <stdio.h> 
//#include <string>
//#include <time.h>
//#include <chrono>     
//#include <thread>         
//#include <fstream>
//using namespace std;
//
////###############################################################################################################################
//// Local Defines
////###############################################################################################################################
//
//#ifndef OPTIMIZED
//#define OPTIMIZEDoff
//#endif
//
//#define cNumberOfBlocks           (40000)
//#define cNumberOfThreadsPerBlock  (64)
//#define cSizeOfDataPerThread      (64)
//#define cNumberOfThreads          (cNumberOfBlocks*cNumberOfThreadsPerBlock)
//
//#ifndef STATUS
//#define CUDA_E_OK                 ((uint8_t)0)
//#define CUDA_NOT_OK               ((uint8_t)1)
//#define CUDA_PENDING_OK           ((uint8_t)2)
//#endif
//
//#ifndef CMD
//#define CMD_OUTPUT_FILE            ((uint8_t)1)
//#define CMD_ADRR_OP                ((uint8_t)2)
//#define CMD_ADRR_MIN               ((uint8_t)3)
//#define CMD_VALID_ZEROES           ((uint8_t)4)
//#define CMD_COUNT                  ((uint8_t)5)
//#endif
//
//
//#define mPRINT_TIME(func)                                                        \
//	auto startTimer = chrono::steady_clock::now();                               \
//	func;                                                                        \
//	auto endTimer = chrono::steady_clock::now();                                 \
//	cout << "Elapsed time in milliseconds : "                                    \
//	<< chrono::duration_cast<chrono::milliseconds>(endTimer - startTimer).count()\
//	<< " ms" << endl;
//
////###############################################################################################################################
//// External Libs for CUDA 
////###############################################################################################################################
///*User libraries for CUDA*/
//#include "DiceCalcCudaTypes.cuh"
//#include "sha3Cuda.cuh"
//#include "SwatchCuda.cuh"
//#include "ValidationCuda.cuh"
//#include "RandomGenCuda.cuh"
//#include "DiceCudaCalculation.cuh"
//#include "FIleWorker.h"
//
////###############################################################################################################################
//// Local Types
////###############################################################################################################################
//
//typedef enum ProgramStates {
//
//	//Prepare Program execution
//	eProgram_Init,
//	eProgram_Get_Console_Options,
//	eProgram_CUDA_Allocate_Memory,
//	eProgram_CUDA_Cpy_Host_Memory,
//	eProgram_CUDA_CURAND_Init,
//
//	//Loop states
//	eProgram_Loop_CUDA_Fill_Random,
//	eProgram_Loop_CUDA_SHA3_Random,
//	eProgram_Loop_Host_Time,
//	eProgram_Loop_CUDA_SHA3_DICE_Proto,
//	eProgram_Loop_CUDA_Validate,
//	eProgram_Loop_Host_Validate,
//	eProgram_Loop_Host_Display_Speed,
//
//	//Prepare to exit
//	eProgram_CUDA_Cpy_Device_Memory,
//	eProgram_Host_Prepare_Check_Unit,
//	eProgram_CUDA_Clean_Device_Memory,
//	eProgram_Exit,
//
//	eProgram_Count,
//}EprogramStates_t;
//
////###############################################################################################################################
//// Local Function Protorypes
////###############################################################################################################################
//
//static void DisplayHeader(void);
//static int writeToFile(const char* outputFile, diceUnitHex_t* diceUnitP);
//
////###############################################################################################################################
//// Local Data
////###############################################################################################################################
//
////Host-CPU
//static cudaDeviceProp props;
//static cudaError_t cudaStatus;
//static EprogramStates_t PStates = eProgram_Init;
//static bool bIsProgramRunning = true;
//static uint8_t aU8Time[sizeof(uint32_t)];
//auto startTimer = chrono::steady_clock::now();
//auto endTimer = chrono::steady_clock::now();
//static size_t sValidDiceUnitIdx = 0;
//static diceUnitHex_t diceUnitValid;
//static char stringBufferL[1024];
//
////Device-GPU
//static payload_t* pD_Payloads = 0;
//static uint8_t* pD_U8Time = 0;
//static diceProto_t* pD_Protos = 0;
//static hashProtoByte_t* pD_ProtosShaHex = 0;
//static bool* pD_ValidatingRes = 0;
//static uint16_t* pD_U16Zeroes = 0;
//
////Copy on Host
//#ifndef OPTIMIZED
//static payload_t h_Payloads[cNumberOfThreads];
//static hashProtoByte_t h_ProtosShaHex[cNumberOfThreads];
//#endif // !OPTIMIZED
//
//static diceProto_t h_Protos[cNumberOfThreads];
//static bool h_ValidatingRes[cNumberOfThreads];
//static uint16_t h_U16Zeroes;
//
////###############################################################################################################################
//// Local Functions
////###############################################################################################################################
//
//int main(int argc, char* argv[])
//{
//	cudaGetDeviceProperties(&props, 0);
//	cudaSetDeviceFlags(cudaDeviceScheduleYield | cudaDeviceMapHost | cudaDeviceLmemResizeToMax);
//
//	//Input from console
//	const char* pOutputFile = 0;
//	const char* pAddrOpL = 0;
//	const char* pAddrMinL = 0;
//	const char* pZeroesL = 0;
//
//	//Set default Proto
//	diceProto_t diceProtoL;
//	payload_t buf_PayloadL;
//
//#ifndef OPTIMIZED
//	int bIsEqualL = CUDA_NOT_OK;
//	uint8_t aShaReturnL[cDICE_SHA3_512_SIZE];
//	uint8_t aShaHexReturnL[cDICE_SHA3_512_SIZE];
//#endif // !OPTIMIZED
//
//	//Get zeroes from string
//	uint8_t aZerosL[cDICE_ZEROES_SIZE];
//
//	while (bIsProgramRunning)
//	{
//		switch (PStates)
//		{
//		case eProgram_Init:
//#ifndef OPTIMIZED
//			//Show Information for GPU
//			DisplayHeader();
//#endif // !OPTIMIZED
//
//			//Reset GPU
//			cudaDeviceReset();
//			cudaThreadExit();
//
//			//Set current GPU Card (zero by Default for single GPU on system)
//			cudaStatus = cudaSetDevice(0);
//
//			//Check for Errors
//			if (cudaStatus != cudaSuccess)
//			{
//				fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//				PStates = eProgram_CUDA_Clean_Device_Memory;
//			}
//			else
//			{
//				PStates = eProgram_Get_Console_Options;
//			}
//			break;
//
//		case eProgram_Get_Console_Options:
//			if (argc == CMD_COUNT)
//			{
//				pOutputFile = argv[CMD_OUTPUT_FILE];
//				pAddrOpL = argv[CMD_ADRR_OP];
//				pAddrMinL = argv[CMD_ADRR_MIN];
//				pZeroesL = argv[CMD_VALID_ZEROES];
//				PStates = eProgram_CUDA_Allocate_Memory;
//			}
//			else
//			{
//				printf("Invalid Arguments\n");
//				PStates = eProgram_CUDA_Clean_Device_Memory;
//			}
//
//			break;
//
//		case eProgram_CUDA_Allocate_Memory:
//			//Set zeroes
//			hexstr_to_char((uint8_t*)pZeroesL, aZerosL, cDICE_ZEROES_SIZE);
//
//			//Allocate memory on GPU
//			cudaStatus = cudaMalloc((void**)&pD_Payloads, cNumberOfThreads * sizeof(payload_t));
//			cudaStatus = cudaMalloc((void**)&pD_Protos, cNumberOfThreads * sizeof(diceProtoHEX_t));
//			cudaStatus = cudaMalloc((void**)&pD_U8Time, sizeof(uint32_t));
//			cudaStatus = cudaMalloc((void**)&pD_ProtosShaHex, cNumberOfThreads * sizeof(hashProtoHex_t));
//			cudaStatus = cudaMalloc((void**)&pD_ValidatingRes, cNumberOfThreads * sizeof(bool));
//			cudaStatus = cudaMalloc((void**)&pD_U16Zeroes, sizeof(uint16_t));
//
//			//Check for Errors
//			if (cudaStatus != cudaSuccess)
//			{
//				fprintf(stderr, "cudaMalloc failed on CUDA_Allocate_Memory!");
//				PStates = eProgram_CUDA_Clean_Device_Memory;
//			}
//			else
//			{
//				PStates = eProgram_CUDA_Cpy_Host_Memory;
//			}
//			break;
//
//		case eProgram_CUDA_Cpy_Host_Memory:
//			//Get seed Time
//			getBeats(aU8Time);
//
//			// Copy output vector from GPU buffer to host memory.
//			cudaStatus = cudaMemcpy(pD_U8Time, aU8Time, sizeof(uint32_t), cudaMemcpyHostToDevice);
//			cudaStatus = cudaMemcpy(pD_U16Zeroes, aZerosL, sizeof(uint8_t), cudaMemcpyHostToDevice);
//
//			//Set const value
//			//HEX -> Byte
//			hexstr_to_char((uint8_t*)pAddrOpL, diceProtoL.addrOp, cDICE_ADDR_SIZE);
//			hexstr_to_char((uint8_t*)pAddrMinL, diceProtoL.addrMin, cDICE_ADDR_SIZE);
//			hexstr_to_char((uint8_t*)pZeroesL, diceProtoL.validZeroes, cDICE_ZEROES_SIZE);
//			//Init with variables
//			memset(diceProtoL.swatchTime, 1, cDICE_SWATCH_TIME_SIZE);
//			memset(diceProtoL.shaPayload, 1, cDICE_SHA3_512_SIZE);
//
//			//Init data in GPU with one CPU
//			for (size_t i = 0; i < cNumberOfThreads; i++)
//			{
//				memcpy(&h_Protos[i], &diceProtoL, cDICE_PROTO_SIZE);
//			}
//
//			cudaStatus = cudaMemcpy(pD_Protos, &h_Protos, sizeof(h_Protos), cudaMemcpyHostToDevice);
//
//			if (cudaStatus != cudaSuccess)
//			{
//				fprintf(stderr, "cudaMemcpy failed on copy Host to Device!");
//				PStates = eProgram_CUDA_Clean_Device_Memory;
//			}
//			else
//			{
//				PStates = eProgram_CUDA_CURAND_Init;
//			}
//			break;
//
//		case eProgram_CUDA_CURAND_Init:
//			// Launch a kernel on the GPU with one thread for each element.
//			gCUDA_CURAND_Init << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_U8Time);
//
//			// Check for any errors launching the kernel
//			cudaStatus = cudaGetLastError();
//			if (cudaStatus != cudaSuccess)
//			{
//				fprintf(stderr, "CURAND_Init launch failed: %s\n", cudaGetErrorString(cudaStatus));
//				PStates = eProgram_CUDA_Clean_Device_Memory;
//			}
//			else
//			{
//				// cudaDeviceSynchronize waits for the kernel to finish, and returns
//				// any errors encountered during the launch.
//				cudaStatus = cudaDeviceSynchronize();
//				if (cudaStatus != cudaSuccess)
//				{
//					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CURAND_Init!\n", cudaStatus);
//					PStates = eProgram_CUDA_Clean_Device_Memory;
//				}
//				else
//				{
//					PStates = eProgram_Loop_CUDA_Fill_Random;
//				}
//			}
//			break;
//
//			//Loop states
//		case eProgram_Loop_CUDA_Fill_Random:
//
//			// Launch a kernel on the GPU with one thread for each element.
//			gCUDA_Fill_Payload << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_Payloads);
//			// Check for any errors launching the kernel
//			cudaStatus = cudaGetLastError();
//			if (cudaStatus != cudaSuccess)
//			{
//				fprintf(stderr, "CUDA_Fill_Payload  launch failed: %s\n", cudaGetErrorString(cudaStatus));
//				PStates = eProgram_CUDA_Clean_Device_Memory;
//			}
//			else
//			{
//				// cudaDeviceSynchronize waits for the kernel to finish, and returns
//				// any errors encountered during the launch.
//				cudaStatus = cudaDeviceSynchronize();
//				if (cudaStatus != cudaSuccess)
//				{
//					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CUDA_Fill_Payload !\n", cudaStatus);
//					PStates = eProgram_CUDA_Clean_Device_Memory;
//				}
//				else
//				{
//#ifndef OPTIMIZED
//					// Copy output vector from GPU buffer to host memory.
//					cudaStatus = cudaMemcpy(h_Payloads, pD_Payloads, cNumberOfThreads * sizeof(payload_t), cudaMemcpyDeviceToHost);
//					if (cudaStatus != cudaSuccess) {
//						fprintf(stderr, "cudaMemcpy failed!");
//						PStates = eProgram_CUDA_Clean_Device_Memory;
//					}
//#endif// !OPTIMIZED
//					PStates = eProgram_Loop_CUDA_SHA3_Random;
//				}
//			}
//			break;
//
//		case eProgram_Loop_CUDA_SHA3_Random:
//			// Launch a kernel on the GPU with one thread for each element.
//			gCUDA_SHA3_Random_Byte << < cNumberOfBlocks, cNumberOfThreadsPerBlock >> > (pD_Payloads, pD_Protos);
//
//			// Check for any errors launching the kernel
//			cudaStatus = cudaGetLastError();
//			if (cudaStatus != cudaSuccess)
//			{
//				fprintf(stderr, "CUDA_SHA3_Random launch failed: %s\n", cudaGetErrorString(cudaStatus));
//				PStates = eProgram_CUDA_Clean_Device_Memory;
//			}
//			else
//			{
//				// cudaDeviceSynchronize waits for the kernel to finish, and returns
//				// any errors encountered during the launch.
//				cudaStatus = cudaDeviceSynchronize();
//				if (cudaStatus != cudaSuccess)
//				{
//					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CUDA_SHA3_Random !\n", cudaStatus);
//					PStates = eProgram_CUDA_Clean_Device_Memory;
//				}
//				else
//				{
//#ifndef OPTIMIZED
//					// Copy output vector from GPU buffer to host memory.
//					cudaStatus = cudaMemcpy(h_Protos, pD_Protos, cNumberOfThreads * sizeof(diceProto_t), cudaMemcpyDeviceToHost);
//					if (cudaStatus != cudaSuccess) {
//						fprintf(stderr, "cudaMemcpy failed!");
//						PStates = eProgram_CUDA_Clean_Device_Memory;
//					}
//#endif// !OPTIMIZED
//					PStates = eProgram_Loop_Host_Time;
//				}
//			}
//			break;
//
//		case eProgram_Loop_Host_Time:
//			//Get seed Time
//			getBeats(aU8Time);
//
//			// Copy output vector from GPU buffer to host memory.
//			cudaStatus = cudaMemcpy(pD_U8Time, aU8Time, sizeof(uint32_t), cudaMemcpyHostToDevice);
//
//			if (cudaStatus != cudaSuccess)
//			{
//				fprintf(stderr, "cudaMemcpy faile on copy Host to Device!");
//				PStates = eProgram_CUDA_Clean_Device_Memory;
//			}
//			else
//			{
//				PStates = eProgram_Loop_CUDA_SHA3_DICE_Proto;
//			}
//
//			break;
//
//		case eProgram_Loop_CUDA_SHA3_DICE_Proto:
//			//Launch a kernel on the GPU with one thread for each element.
//			gCUDA_SHA3_Proto_Byte << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_Protos, pD_U8Time, pD_ProtosShaHex);
//
//			//Check for any errors launching the kernel
//			cudaStatus = cudaGetLastError();
//			if (cudaStatus != cudaSuccess)
//			{
//				fprintf(stderr, "CUDA_SHA3_Proto launch failed: %s\n", cudaGetErrorString(cudaStatus));
//				PStates = eProgram_CUDA_Clean_Device_Memory;
//			}
//			else
//			{
//				//cudaDeviceSynchronize waits for the kernel to finish, and returns
//				//any errors encountered during the launch.
//				cudaStatus = cudaDeviceSynchronize();
//				if (cudaStatus != cudaSuccess)
//				{
//					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CUDA_SHA3_Proto !\n", cudaStatus);
//					PStates = eProgram_CUDA_Clean_Device_Memory;
//				}
//				else
//				{
//					//Copy output vector from GPU buffer to host memory.
//#ifndef OPTIMIZED
//					cudaStatus = cudaMemcpy(h_Protos, pD_Protos, cNumberOfThreads * sizeof(diceProto_t), cudaMemcpyDeviceToHost);
//					cudaStatus = cudaMemcpy(h_ProtosShaHex, pD_ProtosShaHex, cNumberOfThreads * sizeof(hashProtoByte_t), cudaMemcpyDeviceToHost);
//
//					if (cudaStatus != cudaSuccess) {
//						fprintf(stderr, "cudaMemcpy failed!");
//						PStates = eProgram_CUDA_Clean_Device_Memory;
//					}
//#endif// !OPTIMIZED
//					PStates = eProgram_Loop_CUDA_Validate;
//				}
//			}
//			break;
//
//		case eProgram_Loop_CUDA_Validate:
//			//Launch a kernel on the GPU with one thread for each element.
//			gCUDA_ValidateProtoHash_Byte << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_ProtosShaHex, pD_U16Zeroes, pD_ValidatingRes);
//
//			//Check for any errors launching the kernel
//			cudaStatus = cudaGetLastError();
//			if (cudaStatus != cudaSuccess)
//			{
//				fprintf(stderr, "CUDA_ValidateProtoHash launch failed: %s\n", cudaGetErrorString(cudaStatus));
//				PStates = eProgram_CUDA_Clean_Device_Memory;
//			}
//			else
//			{
//				//cudaDeviceSynchronize waits for the kernel to finish, and returns
//				//any errors encountered during the launch.
//				cudaStatus = cudaDeviceSynchronize();
//				if (cudaStatus != cudaSuccess)
//				{
//					fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching CUDA_ValidateProtoHash!\n", cudaStatus);
//					PStates = eProgram_CUDA_Clean_Device_Memory;
//				}
//				else
//				{
//					//Copy output vector from GPU buffer to host memory.
//					cudaStatus = cudaMemcpy(h_ValidatingRes, pD_ValidatingRes, cNumberOfThreads * sizeof(bool), cudaMemcpyDeviceToHost);
//					if (cudaStatus != cudaSuccess) {
//						fprintf(stderr, "cudaMemcpy failed!");
//						PStates = eProgram_CUDA_Clean_Device_Memory;
//					}
//					PStates = eProgram_Loop_Host_Validate;
//				}
//			}
//			break;
//
//		case eProgram_Loop_Host_Validate:
//			PStates = eProgram_Loop_CUDA_Fill_Random;
//			for (size_t i = 0; i < cNumberOfThreads; i++)
//			{
//				if (false == h_ValidatingRes[i])
//				{
//					sValidDiceUnitIdx = i;
//					PStates = eProgram_CUDA_Cpy_Device_Memory;
//					break;
//				}
//			}
//#ifndef OPTIMIZED
//			endTimer = chrono::steady_clock::now();
//			cout << "Elapsed time in milliseconds : "
//				<< chrono::duration_cast<chrono::milliseconds>(endTimer - startTimer).count()
//				<< " ms" << endl;
//			startTimer = chrono::steady_clock::now();
//#endif // !OPTIMIZED
//
//			break;
//
//			//Prepare to exit
//		case eProgram_CUDA_Cpy_Device_Memory:
//			cudaStatus = cudaMemcpy(buf_PayloadL.payload, pD_Payloads[sValidDiceUnitIdx].payload, sizeof(payload_t), cudaMemcpyDeviceToHost);
//			if (cudaStatus != cudaSuccess)
//			{
//				fprintf(stderr, "cudaMemcpy failed!");
//				PStates = eProgram_CUDA_Clean_Device_Memory;
//			}
//			else
//			{
//				PStates = eProgram_Host_Prepare_Check_Unit;
//			}
//			break;
//
//		case eProgram_Host_Prepare_Check_Unit:
//			//Set Up Dice Unit
//			memcpy(diceUnitValid.addrOp, diceProtoL.addrOp, cDICE_ADDR_SIZE);
//			memcpy(diceUnitValid.addrMin, diceProtoL.addrMin, cDICE_ADDR_SIZE);
//			memcpy(diceUnitValid.validZeroes, diceProtoL.validZeroes, cDICE_ZEROES_SIZE);
//			memcpy(diceUnitValid.swatchTime, aU8Time, cDICE_SWATCH_TIME_SIZE);
//
//			//Free up GPU Memory
//			PStates = eProgram_CUDA_Clean_Device_Memory;
//
//#ifndef OPTIMIZED			
//			//Check Hash of proto is as expected
//			//Hash Random
//			sha3_SingleExeuction(diceUnitValid.payload, cDICE_PAYLOAD_SIZE, diceProtoL.shaPayload);
//
//			//Set Time in Global Memory for each Proto
//			memcpy(diceProtoL.swatchTime, diceUnitValid.swatchTime, cDICE_SWATCH_TIME_SIZE);
//
//			//Hash Random
//			sha3_SingleExeuction(&diceProtoL, cDICE_PROTO_SIZE, aShaHexReturnL);
//
//			//Check is the hash value is as expected
//			bIsEqualL = memcmp(aShaHexReturnL, h_ProtosShaHex[sValidDiceUnitIdx].hashProto, cSHA3_512_SIZE);
//
//			if (CUDA_E_OK != bIsEqualL)
//			{
//				PStates = eProgram_Loop_CUDA_Fill_Random;
//			}
//#endif// !OPTIMIZED
//			break;
//
//		case eProgram_CUDA_Clean_Device_Memory:
//			cudaFree(pD_Payloads);
//			cudaFree(pD_Protos);
//			cudaFree(pD_U8Time);
//			cudaFree(pD_ProtosShaHex);
//			cudaFree(pD_ValidatingRes);
//#ifndef OPTIMIZED
//			fprintf(stderr, "Free GPU Memory\n");
//#endif // !OPTIMIZED
//			PStates = eProgram_Exit;
//			break;
//
//		case eProgram_Exit:
//#ifndef OPTIMIZED
//			uint8_t aPrintReadyL[cDICE_UNIT_SIZE + 1];
//			memcpy(aPrintReadyL, h_ProtosShaHex[sValidDiceUnitIdx].hashProto, cDICE_UNIT_SIZE);
//			aPrintReadyL[cDICE_UNIT_SIZE] = '\0';
//
//			printf("Hash of Proto: %s\n", aPrintReadyL);
//#endif // !OPTIMIZED
//
//			writeToFile(pOutputFile, &diceUnitValid);
//
//			bIsProgramRunning = false;
//			break;
//
//		default:
//			bIsProgramRunning = false;
//			fprintf(stderr, "INVALID Program State !!!\n");
//			break;
//		}
//	}
//	return 0;
//}
////###############################################################################################################################
//// CPU - HOST - Functions
////###############################################################################################################################
//
//static void DisplayHeader()
//{
//	const int kb = 1024;
//	const int mb = kb * kb;
//	wcout << "NBody.GPU" << endl << "=========" << endl << endl;
//
//	wcout << "CUDA version:   v" << CUDART_VERSION << endl;
//	//wcout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl;
//
//	int devCount;
//	cudaGetDeviceCount(&devCount);
//	wcout << "CUDA Devices: " << endl << endl;
//
//
//	for (int i = 0; i < devCount; ++i)
//	{
//		cudaDeviceProp props;
//		cudaGetDeviceProperties(&props, i);
//		wcout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
//		wcout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
//		wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
//		wcout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
//		wcout << "  Block registers: " << props.regsPerBlock << endl << endl;
//
//		wcout << "  Warp size:         " << props.warpSize << endl;
//		wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
//		wcout << "  Multiprocessors: " << props.multiProcessorCount << endl;
//		wcout << "  Threads per multiprocessor: " << props.maxThreadsPerMultiProcessor << endl;
//		wcout << "  Concurent kernels: " << props.concurrentKernels << endl;
//		wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << endl;
//		wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << endl;
//		wcout << endl;
//	}
//
//}
//
//static int writeToFile(const char* outputFile, diceUnitHex_t* diceUnitP)
//{
//	int iLenghtL;
//	char addrOp[cDICE_ADDR_SIZE*cBYTE_TO_HEX + 1];
//	char addrMin[cDICE_ADDR_SIZE*cBYTE_TO_HEX + 1];
//	char zeroes[cDICE_ZEROES_SIZE*cBYTE_TO_HEX + 1];
//	char swatchTime[cDICE_SWATCH_TIME_SIZE*cBYTE_TO_HEX + 1];
//	char payload[cDICE_PAYLOAD_SIZE*cBYTE_TO_HEX + 1];
//
//	memcpy(addrOp, diceUnitP->addrOp, cDICE_ADDR_SIZE*cBYTE_TO_HEX);
//	memcpy(addrMin, diceUnitP->addrMin, cDICE_ADDR_SIZE*cBYTE_TO_HEX);
//	memcpy(zeroes, diceUnitP->validZeroes, cDICE_ZEROES_SIZE*cBYTE_TO_HEX);
//	memcpy(swatchTime, diceUnitP->swatchTime, cDICE_SWATCH_TIME_SIZE*cBYTE_TO_HEX);
//	memcpy(payload, diceUnitP->payload, cDICE_PAYLOAD_SIZE*cBYTE_TO_HEX);
//
//	addrOp[cDICE_ADDR_SIZE*cBYTE_TO_HEX] = '\0';
//	addrMin[cDICE_ADDR_SIZE*cBYTE_TO_HEX] = '\0';
//	zeroes[cDICE_ZEROES_SIZE*cBYTE_TO_HEX] = '\0';
//	swatchTime[cDICE_SWATCH_TIME_SIZE*cBYTE_TO_HEX] = '\0';
//	payload[cDICE_PAYLOAD_SIZE*cBYTE_TO_HEX] = '\0';
//
//	iLenghtL = sprintf(stringBufferL, "\{\"addrOperator\": \"%s\",\"addrMiner\" : \"%s\",\"validZeros\" : \"%s\",\"swatchTime\" : \"%s\",	\"payLoad\" : \"%s\" \}", addrOp, addrMin, zeroes, swatchTime, payload);
//
//	ofstream myfile;
//	myfile.open(outputFile);
//	myfile.write(stringBufferL, iLenghtL);
//	myfile.close();
//
//	return 0;
//}
//
////###############################################################################################################################
//// GPU - DEVICE - Functions
////###############################################################################################################################
//
