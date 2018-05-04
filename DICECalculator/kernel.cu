/*
* Copyright (c) 2017, Mihail Maldzhanski <pollarize@gmail.com>
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h> 
#include <string>
#include <time.h>
#include <chrono>           
#include <fstream>
#include <math.h>
using namespace std;

//###############################################################################################################################
// Local Defines
//###############################################################################################################################

#ifndef OPTIMIZED
#define OPTIMIZED
#endif

#ifndef DISPLAY_INFO
#define DISPLAY_INFO
#endif

#define DICE_BYTE

#define cNumberOfBlocks           (4096+1024)
#define cNumberOfThreadsPerBlock  (64)
#define cSizeOfDataPerThread      (64)
#define cNumberOfThreads          (cNumberOfBlocks*cNumberOfThreadsPerBlock)

#ifndef STATUS
#define CUDA_E_OK                 ((uint8_t)0)
#define CUDA_NOT_OK               ((uint8_t)1)
#define CUDA_PENDING_OK           ((uint8_t)2)
#endif

#ifndef CMD
#define CMD_OUTPUT_FILE            ((uint8_t)1)
#define CMD_ADRR_OP                ((uint8_t)2)
#define CMD_ADRR_MIN               ((uint8_t)3)
#define CMD_VALID_ZEROES           ((uint8_t)4)
#define CMD_GLOBAL_TH              ((uint8_t)5)
#define CMD_COUNT                  ((uint8_t)6)
#endif

//Index validation
#define cINVALID_IDX               ((int)-1)

//Print specific
#define cPRINT_SPEED_TH            ((int)20)

//Random Gen Optimizations
#define cRAND_DEV_OK                ((uint8_t)0)
#define cRAND_DEV_TH                ((uint8_t)100)

//256 Trailin zeroes
#define cPZeroes_256                ('0')

//Size of new file name for scrapping
#define cSizeOfStrappingName        ((uint16_t)512)

#define mPRINT_TIME(func)                                                        \
	startTimer = chrono::steady_clock::now();                                    \
	func;                                                                        \
	endTimer = chrono::steady_clock::now();                                      \
	cout << "Elapsed time in milliseconds : "                                    \
	<< chrono::duration_cast<chrono::milliseconds>(endTimer - startTimer).count()\
	<< " ms" << endl;

#define mCUDA_HANDLE_ERROR(func)                                                           \
	cudaStatus = func;                                                                     \
	if (cudaStatus != cudaSuccess)                                                         \
	{                                                                                      \
		fprintf(stderr, "Program Execution Failed: %s\n", cudaGetErrorString(cudaStatus)); \
		PStates = eProgram_CUDA_Clean_Device_Memory;                                       \
	}

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
	eProgram_Loop_Host_Display_Speed,
	eProgram_Loop_Host_Scrapping,

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

static void DisplayHeader(void);

#ifndef DICE_BYTE
static int  writeToFile(const char* outputFile, diceUnitHex_t* diceUnitP);
#else
static int  writeToFile_Byte(const char* outputFile, diceUnit_t* diceUnitP);
#endif // !DICE_BYTE
static void PrintSpeed(void);

//Program Specific
static void Program_Init(void);
static EprogramStates_t Program_GetConsole_Options(int argc, char* argv[]);
static void Program_Allocate_GPU(void);
static void Program_Cpy_Host_To_Device(void);
static void Program_CURAND_Init(void);
static void Program_Fill_Random(int iCycleL);
static void Program_SHA3_Random(void);
static void Program_GetTime(void);
static void Program_SHA3_Proto(void);
static void Program_Validate(void);
static void Program_Host_Validate(void);
static void Program_Cpy_Device_To_Host(uint8_t index);
static void Program_Prepare_Check_Unit(uint8_t index);
static void Program_Clean_Memory(void);
static void Program_Exit(void);

//###############################################################################################################################
// Local Data
//###############################################################################################################################

//Host-CPU
static cudaDeviceProp props;
static cudaError_t cudaStatus;
static EprogramStates_t PStates = eProgram_Init;
static bool bIsProgramRunning = true;
static uint8_t aU8Time[cDICE_SWATCH_TIME_SIZE];
static auto startTimer = chrono::steady_clock::now();
static auto endTimer = chrono::steady_clock::now();
static diceUnit_t diceUnitValid[cBufOut_Size];
static char stringBufferL[1024];
static int iCycles = 0;

//Input from console
static const char* pOutputFile = 0;
static const char* pAddrOpL = 0;
static const char* pAddrMinL = 0;
static const char* pZeroesL = 0;
static const char* pGlobalThL = 0;

//Set default Proto
static diceProto_t diceProtoL;
static payload_t buf_PayloadL;

#ifndef OPTIMIZED
static int bIsEqualL = CUDA_NOT_OK;
static uint8_t aShaReturnL[cDICE_SHA3_512_SIZE];
static uint8_t aShaHexReturnL[cDICE_UNIT_SIZE];
#endif // !OPTIMIZED

//Get zeroes from string
static uint16_t u16Zeroes;
static uint16_t u16GlobalTh;

//Calculate whole execution time
static auto progTimer = chrono::steady_clock::now();

//Device-GPU
static payload_t* pD_Payloads = 0;
static uint8_t* pD_U8Time = 0;
static diceProto_t* pD_Protos = 0;
static hashProto_t* pD_ProtosShaHex = 0;
static uint16_t* pD_U16Zeroes = 0;
static uint16_t* pD_U16GlobalTh = 0;
static int* pD_ValidIDX = 0;

//Copy on Host
#ifndef OPTIMIZED
static payload_t h_Payloads[cNumberOfThreads];
static hashProto_t h_ProtosShaHex[cNumberOfThreads];
static uint16_t h_U16Zeroes;
#endif // !OPTIMIZED

static diceProto_t h_Protos[cNumberOfThreads];
static int h_ValidIDX[cBufOut_Size] = {cINVALID_IDX,cINVALID_IDX};
static char aSrappingName[cSizeOfStrappingName];
//###############################################################################################################################
// Local Functions
//###############################################################################################################################

int main(int argc, char* argv[])
{
	while (bIsProgramRunning)
	{
		switch (PStates)
		{
		case eProgram_Init:
			//Set default next state
			PStates = eProgram_Get_Console_Options;

			//Execute program
			Program_Init();
			break;

		case eProgram_Get_Console_Options:
			//Set default next state
			PStates = Program_GetConsole_Options(argc, argv);
			break;

		case eProgram_CUDA_Allocate_Memory:
			//Set default next state
			PStates = eProgram_CUDA_Cpy_Host_Memory;

			//Execute PRogram specific
			Program_Allocate_GPU();
			break;

		case eProgram_CUDA_Cpy_Host_Memory:
			//Set default next state
			PStates = eProgram_CUDA_CURAND_Init;

			//Execute
			Program_Cpy_Host_To_Device();
			break;

		case eProgram_CUDA_CURAND_Init:
			//Set default next state
			PStates = eProgram_Loop_CUDA_Fill_Random;

			//Execute
			Program_CURAND_Init();
			break;


			//Loop states
		case eProgram_Loop_CUDA_Fill_Random:
			//Set default next state
			PStates = eProgram_Loop_CUDA_SHA3_Random;

			//Execute
			Program_Fill_Random(iCycles);
			break;

		case eProgram_Loop_CUDA_SHA3_Random:
			//Set default next state
			PStates = eProgram_Loop_Host_Time;

			//Execute
			Program_SHA3_Random();
			break;

		case eProgram_Loop_Host_Time:
			//Set default next state
			PStates = eProgram_Loop_CUDA_SHA3_DICE_Proto;

			//Execute
			Program_GetTime();
			break;

		case eProgram_Loop_CUDA_SHA3_DICE_Proto:
			//Set default next state
			PStates = eProgram_Loop_CUDA_Validate;

			//Execute
			Program_SHA3_Proto();
			break;

		case eProgram_Loop_CUDA_Validate:
			//Set default next state
			PStates = eProgram_Loop_Host_Validate;

			//Execute
			Program_Validate();
			break;

		case eProgram_Loop_Host_Validate:
			//Set default next state
			PStates = eProgram_Loop_CUDA_Fill_Random;

			//Execute
			Program_Host_Validate();
			break;

		case eProgram_Loop_Host_Scrapping:
			//Set default next state
			PStates = eProgram_Loop_CUDA_Fill_Random;

			//Execute
			Program_Cpy_Device_To_Host(cBufOut_Scrappting);
			Program_Prepare_Check_Unit(cBufOut_Scrappting);
			snprintf(aSrappingName, cSizeOfStrappingName, "%s%d%d%d%d_%d.dicescr", pOutputFile, aU8Time[0], aU8Time[1], aU8Time[2], aU8Time[3],iCycles);
			writeToFile_Byte(aSrappingName, &diceUnitValid[cBufOut_Scrappting]);
			break;


			//Prepare to exit
		case eProgram_CUDA_Cpy_Device_Memory:
			//Set default next state
			PStates = eProgram_Host_Prepare_Check_Unit;

			//Execute
			Program_Cpy_Device_To_Host(cBufOut_General);
			break;

		case eProgram_Host_Prepare_Check_Unit:
			//Free up GPU Memory
			PStates = eProgram_CUDA_Clean_Device_Memory;

			//Execute
			Program_Prepare_Check_Unit(cBufOut_General);
			break;

		case eProgram_CUDA_Clean_Device_Memory:
			//Set default next state
			PStates = eProgram_Exit;

			//Execute
			Program_Clean_Memory();
			break;

		case eProgram_Exit:
			Program_Exit();
			break;

		default:
			bIsProgramRunning = false;
			fprintf(stderr, "INVALID Program State !!!\n");
			break;
		}
	}

	//Print spent time in s
	endTimer = chrono::steady_clock::now();

	cout << "Time used: "
		<< (chrono::duration_cast<chrono::seconds>(endTimer - progTimer).count())
		<< " s" << endl;

	//Exit from program
	return 0;
}

//###############################################################################################################################
// CPU - HOST - Functions
//###############################################################################################################################

static void DisplayHeader()
{
#ifdef DISPLAY_INFO

	const int kb = 1024;
	const int mb = kb * kb;
	int devCount;

	cudaGetDeviceCount(&devCount);
	wcout << "CUDA Devices: " << endl << endl;

	wcout << "NBody.GPU" << endl << "=========" << endl << endl;
	wcout << "CUDA version:   v" << CUDART_VERSION << endl;

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
		wcout << "  Concurrent kernels: " << props.concurrentKernels << endl;
		wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << endl;
		wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << endl;
		wcout << endl;
	}
#endif // !DISPLAY_INFO

	//Info for starting 
	printf("CUDA DICE Calculator has been started\n");
	printf("Blocks: %d Threads per block: %d Total Threads: %d \n", cNumberOfBlocks, cNumberOfThreadsPerBlock, cNumberOfThreads);

}

#ifndef DICE_BYTE
static int writeToFile(const char* outputFile, diceUnitHex_t* diceUnitP)
{
	int iLenghtL;
	char addrOp[cDICE_ADDR_SIZE*cBYTE_TO_HEX + 1];
	char addrMin[cDICE_ADDR_SIZE*cBYTE_TO_HEX + 1];
	char zeroes[cDICE_ZEROES_SIZE*cBYTE_TO_HEX + 1];
	char swatchTime[cDICE_SWATCH_TIME_SIZE*cBYTE_TO_HEX + 1];
	char payload[cDICE_PAYLOAD_SIZE*cBYTE_TO_HEX + 1];

	memcpy(addrOp, diceUnitP->addrOp, cDICE_ADDR_SIZE*cBYTE_TO_HEX);
	memcpy(addrMin, diceUnitP->addrMin, cDICE_ADDR_SIZE*cBYTE_TO_HEX);
	memcpy(zeroes, diceUnitP->validZeroes, cDICE_ZEROES_SIZE*cBYTE_TO_HEX);
	memcpy(swatchTime, diceUnitP->swatchTime, cDICE_SWATCH_TIME_SIZE*cBYTE_TO_HEX);
	memcpy(payload, diceUnitP->payload, cDICE_PAYLOAD_SIZE*cBYTE_TO_HEX);

	addrOp[cDICE_ADDR_SIZE*cBYTE_TO_HEX] = '\0';
	addrMin[cDICE_ADDR_SIZE*cBYTE_TO_HEX] = '\0';
	zeroes[cDICE_ZEROES_SIZE*cBYTE_TO_HEX] = '\0';
	swatchTime[cDICE_SWATCH_TIME_SIZE*cBYTE_TO_HEX] = '\0';
	payload[cDICE_PAYLOAD_SIZE*cBYTE_TO_HEX] = '\0';

	iLenghtL = sprintf(stringBufferL, "{\"addrOperator\": \"%s\",\"addrMiner\" : \"%s\",\"validZeros\" : \"%s\",\"swatchTime\" : \"%s\",	\"payLoad\" : \"%s\" }", addrOp, addrMin, zeroes, swatchTime, payload);

	ofstream myfile;
	myfile.open(outputFile);
	myfile.write(stringBufferL, iLenghtL);
	myfile.close();

	return 0;
}
#else
static int writeToFile_Byte(const char* outputFile, diceUnit_t* diceUnitP)
{
	int iLenghtL;
	char addrOp[cDICE_ADDR_SIZE*(cBYTE_TO_HEX + cBYTE_TO_HEX) + 1];
	char addrMin[cDICE_ADDR_SIZE*(cBYTE_TO_HEX + cBYTE_TO_HEX) + 1];
	char zeroes[cDICE_ZEROES_SIZE*(cBYTE_TO_HEX + cBYTE_TO_HEX) + 1];
	char swatchTime[cDICE_SWATCH_TIME_SIZE*(cBYTE_TO_HEX + cBYTE_TO_HEX) + 1];
	char payload[cDICE_PAYLOAD_SIZE*(cBYTE_TO_HEX + cBYTE_TO_HEX) + 1];

	dCUDA_Char_To_HexStr(diceUnitP->addrOp, cDICE_ADDR_SIZE*cBYTE_TO_HEX, (uint8_t*)addrOp);
	dCUDA_Char_To_HexStr(diceUnitP->addrMin, cDICE_ADDR_SIZE*cBYTE_TO_HEX, (uint8_t*)addrMin);
	dCUDA_Char_To_HexStr(diceUnitP->validZeroes, cDICE_ZEROES_SIZE*cBYTE_TO_HEX, (uint8_t*)zeroes);
	dCUDA_Char_To_HexStr(diceUnitP->swatchTime, cDICE_SWATCH_TIME_SIZE*cBYTE_TO_HEX, (uint8_t*)swatchTime);
	dCUDA_Char_To_HexStr(diceUnitP->payload, cDICE_PAYLOAD_SIZE*cBYTE_TO_HEX, (uint8_t*)payload);

	addrOp[cDICE_ADDR_SIZE*(cBYTE_TO_HEX + cBYTE_TO_HEX)] = '\0';
	addrMin[cDICE_ADDR_SIZE*(cBYTE_TO_HEX + cBYTE_TO_HEX)] = '\0';
	zeroes[cDICE_ZEROES_SIZE*(cBYTE_TO_HEX + cBYTE_TO_HEX)] = '\0';
	swatchTime[cDICE_SWATCH_TIME_SIZE*(cBYTE_TO_HEX + cBYTE_TO_HEX)] = '\0';
	payload[cDICE_PAYLOAD_SIZE*(cBYTE_TO_HEX + cBYTE_TO_HEX)] = '\0';

	iLenghtL = sprintf(stringBufferL, "{\"addrOperator\": \"%s\",\"addrMiner\" : \"%s\",\"validZeros\" : \"%s\",\"swatchTime\" : \"%s\",	\"payLoad\" : \"%s\" }", addrOp, addrMin, zeroes, swatchTime, payload);

	ofstream myfile;
	myfile.open(outputFile);
	myfile.write(stringBufferL, iLenghtL);
	myfile.close();

	return 0;
}
#endif // !DICE_BYTE

static void PrintSpeed(void)
{
	uint64_t s64SpeedL = 0;
	size_t   sCountOfMaxOpL = 0;
	endTimer = chrono::steady_clock::now();
	if (iCycles == cPRINT_SPEED_TH)
	{
		s64SpeedL = chrono::duration_cast<chrono::milliseconds> (endTimer - startTimer).count();
		s64SpeedL = 1000 / s64SpeedL; //in seconds
		s64SpeedL = s64SpeedL * cNumberOfThreads;
		sCountOfMaxOpL =(size_t) pow(2, u16Zeroes);
		cout << "Operations per second: "<< s64SpeedL << " / s" << endl;
		cout << "Estimated time for generation (max): "<< sCountOfMaxOpL / s64SpeedL << " s" << endl;
	}
	startTimer = chrono::steady_clock::now();
}

static void Program_Init(void)
{
	//Show Information for GPU
	DisplayHeader();

	//Reset GPU
	cudaDeviceReset();
	cudaThreadExit();
	cudaGetDeviceProperties(&props, 0);
	cudaSetDeviceFlags(cudaDeviceScheduleYield | cudaDeviceMapHost | cudaDeviceLmemResizeToMax);

	//Set current GPU Card (zero by Default for single GPU on system)
	mCUDA_HANDLE_ERROR(cudaSetDevice(0));
}

static EprogramStates_t Program_GetConsole_Options(int argc, char* argv[])
{
	EprogramStates_t ProgramStateL = eProgram_CUDA_Allocate_Memory;
	if (argc == CMD_COUNT)
	{
		pOutputFile = argv[CMD_OUTPUT_FILE];
		pAddrOpL = argv[CMD_ADRR_OP];
		pAddrMinL = argv[CMD_ADRR_MIN];
		pZeroesL = argv[CMD_VALID_ZEROES];
		pGlobalThL = argv[CMD_GLOBAL_TH];
	}
	else
	{
		printf("Invalid Arguments\n");
		ProgramStateL = eProgram_CUDA_Clean_Device_Memory;
	}
	return ProgramStateL;
}

static void Program_Allocate_GPU(void)
{
	//Set zeroes
	if (cPZeroes_256 != *pZeroesL)
	{
		hexstr_to_char((uint8_t*)pZeroesL,(uint8_t*)&u16Zeroes, cDICE_ZEROES_SIZE);
		hexstr_to_char((uint8_t*)pGlobalThL, (uint8_t*)&u16GlobalTh, cDICE_ZEROES_SIZE);
	}
	else
	{
		u16Zeroes = 256;
		u16GlobalTh = 256;
	}

	//Allocate memory on GPU
	mCUDA_HANDLE_ERROR(cudaMalloc((void**)&pD_Payloads, cNumberOfThreads * sizeof(payload_t)));
	mCUDA_HANDLE_ERROR(cudaMalloc((void**)&pD_Protos, cNumberOfThreads * sizeof(diceProto_t)));
	mCUDA_HANDLE_ERROR(cudaMalloc((void**)&pD_ProtosShaHex, cNumberOfThreads * sizeof(hashProto_t)));
	mCUDA_HANDLE_ERROR(cudaMalloc((void**)&pD_U16Zeroes, sizeof(uint16_t)));
	mCUDA_HANDLE_ERROR(cudaMalloc((void**)&pD_U16GlobalTh, sizeof(uint16_t)));
	mCUDA_HANDLE_ERROR(cudaMalloc((void**)&pD_U8Time, cDICE_SWATCH_TIME_SIZE));
	mCUDA_HANDLE_ERROR(cudaMalloc((void**)&pD_ValidIDX, sizeof(int)*cBufOut_Size));
}

static void Program_Cpy_Host_To_Device(void)
{
	//Get seed Time
	getBeats(aU8Time);

	// Copy data from Host to Device
	mCUDA_HANDLE_ERROR(cudaMemcpy(pD_U8Time, aU8Time, cDICE_SWATCH_TIME_SIZE, cudaMemcpyHostToDevice));
	mCUDA_HANDLE_ERROR(cudaMemcpy(pD_U16Zeroes, &u16Zeroes, sizeof(uint16_t), cudaMemcpyHostToDevice));
	mCUDA_HANDLE_ERROR(cudaMemcpy(pD_U16GlobalTh, &u16GlobalTh, sizeof(uint16_t), cudaMemcpyHostToDevice));
	mCUDA_HANDLE_ERROR(cudaMemset(pD_ValidIDX, cINVALID_IDX, sizeof(int)*cBufOut_Size));

	//Set const value
#ifndef DICE_BYTE
	memcpy(diceProtoL.addrMin, pAddrMinL, cDICE_ADDR_SIZE * cBYTE_TO_HEX);
	memcpy(diceProtoL.addrOp, pAddrOpL, cDICE_ADDR_SIZE * cBYTE_TO_HEX);
	memcpy(diceProtoL.validZeroes, pZeroesL, cDICE_ZEROES_SIZE * cBYTE_TO_HEX);
#else
	//HEX -> Byte
	hexstr_to_char((uint8_t*)pAddrOpL, diceProtoL.addrOp, cDICE_ADDR_SIZE);
	hexstr_to_char((uint8_t*)pAddrMinL, diceProtoL.addrMin, cDICE_ADDR_SIZE);
	hexstr_to_char((uint8_t*)pZeroesL, diceProtoL.validZeroes, cDICE_ZEROES_SIZE);
#endif // !DICE_BYTE

	memset(diceProtoL.swatchTime, 1, cDICE_SWATCH_TIME_SIZE * cBYTE_TO_HEX);
	memset(diceProtoL.shaPayload, 1, cDICE_SHA3_512_SIZE * cBYTE_TO_HEX);

	//Init data in GPU with one CPU
	for (size_t i = 0; i < cNumberOfThreads; i++)
	{
		memcpy(&h_Protos[i], &diceProtoL, cDICE_PROTO_SIZE * cBYTE_TO_HEX);
	}

	//Copy to GPU
	mCUDA_HANDLE_ERROR(cudaMemcpy(pD_Protos, &h_Protos, sizeof(h_Protos), cudaMemcpyHostToDevice));
}

static void Program_CURAND_Init(void)
{
	// Launch a kernel on the GPU with one thread for each element.
	gCUDA_CURAND_Init << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_U8Time);

	// Check for any errors launching the kernel
	mCUDA_HANDLE_ERROR(cudaGetLastError());
	mCUDA_HANDLE_ERROR(cudaDeviceSynchronize());
}

static void Program_Fill_Random(int iCycleL)
{
	if (cRAND_DEV_OK == (iCycleL%cRAND_DEV_TH))
	{
		// Launch a kernel on the GPU with one thread for each element.
		gCUDA_Fill_Payload << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_Payloads);
	}
	else
	{
		// Launch a kernel on the GPU with one thread for each element.
		gCUDA_Fill_Payload_Four_Byte << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_Payloads);
	}

	// Check for any errors launching the kernel
	mCUDA_HANDLE_ERROR(cudaGetLastError());
	mCUDA_HANDLE_ERROR(cudaDeviceSynchronize());
#ifndef OPTIMIZED
	// Copy output vector from GPU buffer to host memory.
	mCUDA_HANDLE_ERROR(cudaMemcpy(h_Payloads, pD_Payloads, cNumberOfThreads * sizeof(payload_t), cudaMemcpyDeviceToHost));
#endif// !OPTIMIZED
}

static void Program_SHA3_Random(void)
{
	// Launch a kernel on the GPU with one thread for each element.
#ifndef DICE_BYTE
	gCUDA_SHA3_Random << < cNumberOfBlocks, cNumberOfThreadsPerBlock >> > (pD_Payloads, pD_Protos);
#else
	gCUDA_SHA3_Random_Byte << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_Payloads, pD_Protos);
#endif // !DICE_BYTE

	// Check for any errors launching the kernel
	mCUDA_HANDLE_ERROR(cudaGetLastError());
	mCUDA_HANDLE_ERROR(cudaDeviceSynchronize());
#ifndef OPTIMIZED
	// Copy output vector from GPU buffer to host memory.
	mCUDA_HANDLE_ERROR(cudaMemcpy(h_Protos, pD_Protos, cNumberOfThreads * sizeof(diceProto_t), cudaMemcpyDeviceToHost));
#endif// !OPTIMIZED
}

static void Program_GetTime(void)
{
	//Get Time
	getBeats(aU8Time);

	// Copy data
	mCUDA_HANDLE_ERROR(cudaMemcpy(pD_U8Time, aU8Time, cDICE_SWATCH_TIME_SIZE, cudaMemcpyHostToDevice));
}

static void Program_SHA3_Proto(void)
{

	//Launch a kernel on the GPU with one thread for each element.
#ifndef DICE_BYTE
	gCUDA_SHA3_Proto << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_Protos, pD_U8Time, pD_ProtosShaHex);
#else
	gCUDA_SHA3_Proto_Byte << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_Protos, pD_U8Time, pD_ProtosShaHex);
#endif // !DICE_BYTE

	// Check for any errors launching the kernel
	mCUDA_HANDLE_ERROR(cudaGetLastError());
	mCUDA_HANDLE_ERROR(cudaDeviceSynchronize());
#ifndef OPTIMIZED
	//Copy output vector from GPU buffer to host memory.
	mCUDA_HANDLE_ERROR(cudaMemcpy(h_Protos, pD_Protos, cNumberOfThreads * sizeof(diceProto_t), cudaMemcpyDeviceToHost));
	mCUDA_HANDLE_ERROR(cudaMemcpy(h_ProtosShaHex, pD_ProtosShaHex, cNumberOfThreads * sizeof(hashProto_t), cudaMemcpyDeviceToHost));
#endif// !OPTIMIZED
}

static void Program_Validate(void)
{
	//Launch a kernel on the GPU with one thread for each element.
#ifndef DICE_BYTE
	gCUDA_ValidateProtoHash << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_ProtosShaHex, pD_U16Zeroes, pD_ValidIDX);
#else
	gCUDA_ValidateProtoHash_Byte << < cNumberOfBlocks, cNumberOfThreadsPerBlock, cSizeOfDataPerThread >> > (pD_ProtosShaHex, pD_U16Zeroes, pD_U16GlobalTh, pD_ValidIDX);
#endif // !DICE_BYTE

	// Check for any errors launching the kernel
	mCUDA_HANDLE_ERROR(cudaGetLastError());
	mCUDA_HANDLE_ERROR(cudaDeviceSynchronize());

	//Copy output vector from GPU buffer to host memory.
	mCUDA_HANDLE_ERROR(cudaMemcpy(&h_ValidIDX, pD_ValidIDX, sizeof(int)*cBufOut_Size, cudaMemcpyDeviceToHost));
	mCUDA_HANDLE_ERROR(cudaMemset(pD_ValidIDX, cINVALID_IDX, sizeof(int)*cBufOut_Size));
}

static void Program_Host_Validate(void)
{
	iCycles++;
	if (cINVALID_IDX != h_ValidIDX[cBufOut_General])
	{
		PStates = eProgram_CUDA_Cpy_Device_Memory;
	}
	else if (cINVALID_IDX != h_ValidIDX[cBufOut_Scrappting])
	{
		PStates = eProgram_Loop_Host_Scrapping;
	}
	else
	{
		//Notging
	}

	//Print count of Operations per second
	PrintSpeed();
}

static void Program_Cpy_Device_To_Host(uint8_t index)
{
	mCUDA_HANDLE_ERROR(cudaMemcpy(buf_PayloadL.payload, pD_Payloads[h_ValidIDX[index]].payload, cDICE_PAYLOAD_SIZE, cudaMemcpyDeviceToHost));
}

static void Program_Prepare_Check_Unit(uint8_t index)
{
	//Set Up Dice Unit
	memcpy(diceUnitValid[index].addrOp, diceProtoL.addrOp, cDICE_ADDR_SIZE * cBYTE_TO_HEX);
	memcpy(diceUnitValid[index].addrMin, diceProtoL.addrMin, cDICE_ADDR_SIZE * cBYTE_TO_HEX);
	memcpy(diceUnitValid[index].validZeroes, diceProtoL.validZeroes, cDICE_ZEROES_SIZE * cBYTE_TO_HEX);
#ifndef DICE_BYTE
	dCUDA_Char_To_HexStr(aU8Time, cDICE_SWATCH_TIME_SIZE, diceUnitValid.swatchTime);
	dCUDA_Char_To_HexStr(buf_PayloadL.payload, sizeof(payload_t), diceUnitValid.payload);
#else
	memcpy(diceUnitValid[index].swatchTime, aU8Time, cDICE_SWATCH_TIME_SIZE);
	memcpy(diceUnitValid[index].payload, buf_PayloadL.payload, cDICE_PAYLOAD_SIZE);
#endif // !DICE_BYTE

#ifndef OPTIMIZED 
#ifndef	DICE_BYTE	
	//Check Hash of proto is as expected
	//Hash Random
	sha3_SingleExeuction(diceUnitValid.payload, cDICE_PAYLOAD_SIZE * cBYTE_TO_HEX, aShaReturnL);

	//Save data to Global Memory in HexString
	dCUDA_Char_To_HexStr(aShaReturnL, cDICE_SHA3_512_SIZE, diceProtoL.shaPayload);

	//Set Time in Global Memory for each Proto
	memcpy(diceProtoL.swatchTime, diceUnitValid.swatchTime, cDICE_SWATCH_TIME_SIZE * cBYTE_TO_HEX);

	//Hash Random
	sha3_SingleExeuction(&diceProtoL, cDICE_PROTO_SIZE * cBYTE_TO_HEX, aShaReturnL);

	//Save data to Global Memory in HexString 
	dCUDA_Char_To_HexStr(aShaReturnL, cDICE_SHA3_512_SIZE, aShaHexReturnL);

	//Check is the hash value is as expected
	bIsEqualL = memcmp(aShaHexReturnL, h_ProtosShaHex[sValidDiceUnitIdx].hashProto, cDICE_UNIT_SIZE);

	if (CUDA_E_OK != bIsEqualL)
	{
		PStates = eProgram_Loop_CUDA_Fill_Random;
	}
#endif// !DICE_BYTE
#endif// !OPTIMIZED
}

static void Program_Clean_Memory(void)
{
	//Free GPU Memory
	cudaFree(pD_Payloads);
	cudaFree(pD_Protos);
	cudaFree(pD_U8Time);
	cudaFree(pD_ProtosShaHex);
	cudaFree(pD_U16Zeroes);
	cudaFree(pD_U16GlobalTh);
	cudaFree(pD_ValidIDX);
#ifndef OPTIMIZED
	fprintf(stderr, "Free GPU Memory\n");
#endif // !OPTIMIZED
}

static void Program_Exit(void)
{
#ifndef OPTIMIZED
#ifndef	DICE_BYTE	
	uint8_t aPrintReadyL[(cSHA3_512_SIZE*cBYTE_TO_HEX) + 1];
	memcpy(aPrintReadyL, h_ProtosShaHex[h_ValidIDX].hashProto, (cSHA3_512_SIZE*cBYTE_TO_HEX));
	aPrintReadyL[(cSHA3_512_SIZE*cBYTE_TO_HEX)] = '\0';

#else
	uint8_t aPrintReadyL[(cSHA3_512_SIZE*(cBYTE_TO_HEX + cBYTE_TO_HEX)) + 1];
	dCUDA_Char_To_HexStr(h_ProtosShaHex[h_ValidIDX[cBufOut_General]].hashProto, cDICE_SHA3_512_SIZE, aPrintReadyL);
	aPrintReadyL[(cSHA3_512_SIZE*(cBYTE_TO_HEX + cBYTE_TO_HEX))] = '\0';
#endif// !DICE_BYTE
	printf("Hash of Proto: %s\n", aPrintReadyL);
#endif // !OPTIMIZED

	//Save to JSON formatted file
#ifndef	DICE_BYTE
	writeToFile(pOutputFile, &diceUnitValid);
#else
	writeToFile_Byte(pOutputFile, &diceUnitValid[cBufOut_General]);
#endif// !DICE_BYTE
	bIsProgramRunning = false;
}
//###############################################################################################################################
// GPU - DEVICE - Functions
//###############################################################################################################################

