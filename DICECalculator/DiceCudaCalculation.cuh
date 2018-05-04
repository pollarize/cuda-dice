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

#define cBufOut_General    ((uint8_t)0)
#define cBufOut_Scrappting ((uint8_t)1)

#define cBufOut_Size       ((uint8_t)2)

#define mMinMarginFromGlobalTh(globalTh)    (uint16_t)((globalTh)-(uint16_t)10)
//###############################################################################################################################
// Local Data on GPU
//###############################################################################################################################

//###############################################################################################################################
// External Function Prototypes
//###############################################################################################################################

 __device__ __host__ uint8_t* dCUDA_Char_To_HexStr(uint8_t* pCharArrayP, uint8_t u8CountOfBytesP, uint8_t* bufOut);
 __device__ __host__ void hexstr_to_char(uint8_t* hexstr, uint8_t* bufOut, uint8_t size);
 //###############################################################################################################################
 // GPU-Kernals - HEX
 //###############################################################################################################################

__global__ void gCUDA_SHA3_Random(payload_t* bufIn, diceProto_t* bufOut)
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

__global__ void gCUDA_SHA3_Proto(diceProto_t* bufIn, uint8_t* bufTime, uint16_t* globalTh,hashProto_t* bufOut)
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

__global__ void gCUDA_ValidateProtoHash(hashProto_t* bufIn, uint16_t* zeroes , uint16_t* globalTh, int* bufOut)
{
	int idx = (blockDim.x*blockIdx.x) + threadIdx.x;
	if (idx < cNumberOfThreads)
	{
		//Convert Hex String to Byte Array
		uint8_t aShaReturnL[64];
		hexstr_to_char(bufIn[idx].hashProto, aShaReturnL,64);

		bool localRes = true;
		bool globalRes = true;

		validateHash(*zeroes, bufIn[idx].hashProto, &localRes);
		validateHash(mMinMarginFromGlobalTh(*globalTh), bufIn[idx].hashProto, &globalRes);

		if (true != localRes)
		{
			bufOut[cBufOut_General] = idx;
		}
		else if (true != globalRes)
		{
			bufOut[cBufOut_Scrappting] = idx;
		}
		else
		{
			//Invalid do NOTHING
		}
	}
}

//###############################################################################################################################
// GPU-Kernals - Byte
//###############################################################################################################################

__global__ void gCUDA_SHA3_Random_Byte(payload_t* bufIn, diceProto_t* bufOut)
{
	int idx = (blockDim.x*blockIdx.x) + threadIdx.x;
	if (idx < cNumberOfThreads)
	{
		//Hash Random Bytes
		sha3_SingleExeuction(bufIn[idx].payload, cDICE_PAYLOAD_SIZE, bufOut[idx].shaPayload);
	}
}

__global__ void gCUDA_SHA3_Proto_Byte(diceProto_t* bufIn, uint8_t* bufTime, hashProto_t* bufOut)
{
	int idx = (blockDim.x*blockIdx.x) + threadIdx.x;
	if (idx < cNumberOfThreads)
	{
		//Set Time in Global Memory for each Proto
		memcpy(bufIn[idx].swatchTime,bufTime,cDICE_SWATCH_TIME_SIZE);

		//Hash Random
		sha3_SingleExeuction(&bufIn[idx], 109, bufOut[idx].hashProto);
	}
}

__global__ void gCUDA_ValidateProtoHash_Byte(hashProto_t* bufIn, uint16_t* zeroes, uint16_t* globalTh, int* bufOut)
{
	int idx = (blockDim.x*blockIdx.x) + threadIdx.x;
	if (idx < cNumberOfThreads)
	{
		bool localRes = true;
		bool globalRes = true;

		validateHash(*zeroes, bufIn[idx].hashProto, &localRes);
		validateHash(mMinMarginFromGlobalTh(*globalTh), bufIn[idx].hashProto, &globalRes);

		if (true != localRes)
		{
			bufOut[cBufOut_General] = idx;
		}
		else if (true != globalRes)
		{
			bufOut[cBufOut_Scrappting] = idx;
		}
		else
		{
			//Invalid do NOTHING
		}
	}
}
//###############################################################################################################################
// Local Functions
//###############################################################################################################################

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
