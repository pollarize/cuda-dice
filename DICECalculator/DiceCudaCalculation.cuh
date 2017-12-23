#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string>
#include <curand_kernel.h>
#include "DiceCalcCudaTypes.cuh"
#include "sha3Cuda.cuh"
#include "ValidationCuda.cuh"

#ifndef cNumberOfThreads
#error "There is not defined count of used threads!"
#endif

 __device__ __host__ uint8_t* dCUDA_Char_To_HexStr(uint8_t* pCharArrayP, uint8_t u8CountOfBytesP, uint8_t* bufOut);
 __device__ __host__ void hexstr_to_char(uint8_t* hexstr, uint8_t* bufOut, uint8_t size);

__global__ void gCUDA_SHA3_Random(payload_t* bufIn, diceProtoHEX_t* bufOut)
{
	int idx = (blockDim.x*blockIdx.x) + threadIdx.x;
	if (idx < cNumberOfThreads)
	{
		//Random to Hex String
		uint8_t aHexOfRandomL[83 * 2];
		dCUDA_Char_To_HexStr(bufIn[idx].payload, 83, aHexOfRandomL);

		//Hash Random
		uint8_t aShaReturnL[64];
		sha3_SingleExeuction(aHexOfRandomL, 83 * 2, aShaReturnL);

		//Save data to Global Memory in HexString
		dCUDA_Char_To_HexStr(aShaReturnL, 64, bufOut[idx].shaPayload);
	}
}

__global__ void gCUDA_SHA3_Proto(diceProtoHEX_t* bufIn, uint8_t* bufTime, hashProtoHex_t* bufOut)
{
	int idx = (blockDim.x*blockIdx.x) + threadIdx.x;
	if (idx < cNumberOfThreads)
	{
		//Set Time in Global Memory for each Proto
		dCUDA_Char_To_HexStr(bufTime, 4, bufIn[idx].swatchTime);

		//Hash Random
		uint8_t aShaReturnL[64];
		sha3_SingleExeuction(&bufIn[idx], 109 * 2, aShaReturnL);

		//Save data to Global Memory in HexString
		dCUDA_Char_To_HexStr(aShaReturnL, 64, bufOut[idx].hashProto);
	}
}

__global__ void gCUDA_ValidateProtoHash(hashProtoHex_t* bufIn, uint16_t* zeroes ,bool* bufOut)
{
	int idx = (blockDim.x*blockIdx.x) + threadIdx.x;
	if (idx < cNumberOfThreads)
	{
		//Convert Hex String to Byte Array
		uint8_t aShaReturnL[64];
		hexstr_to_char(bufIn[idx].hashProto, aShaReturnL,64);

		validateHash(*zeroes, aShaReturnL, &bufOut[idx]);
	}
}

 __device__ __host__ void hexstr_to_char(uint8_t* hexstr, uint8_t* bufOut, uint8_t size)
{
	for (size_t i = 0, j = 0; j < size; i += 2, j++)
	{
		bufOut[j] = (hexstr[i] % 32 + 9) % 25 * 16 + (hexstr[i + 1] % 32 + 9) % 25;
	}
}

 __device__ __host__ uint8_t* dCUDA_Char_To_HexStr(uint8_t* pCharArrayP, uint8_t u8CountOfBytesP, uint8_t* bufOut)
{
	const char* aHexL = "0123456789abcdef";
	for (size_t i = 0; i < u8CountOfBytesP; i++)
	{
		bufOut[(i * 2)] = aHexL[(pCharArrayP[i] >> 4) & 0xF];
		bufOut[(i * 2) + 1] = aHexL[pCharArrayP[i] & 0xF];
	}
	return bufOut;
}