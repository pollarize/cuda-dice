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
#include <curand_kernel.h>
#include "DiceCalcCudaTypes.cuh"

#ifndef cNumberOfThreads
#error "There is not defined count of used threads!"
#endif

#define cPayloadSize    ((uint8_t)83)
#define cU32Size        ((uint8_t)4)  
#define cPayloadSizeU32 ((uint8_t)80)

__device__ curandState_t CurandStateHolder[cNumberOfThreads];

__host__ void getPayload(uint8_t* buffer, int size)
{
	srand((unsigned int)time(NULL));

	for (size_t i = 0; i < size; i++)
	{
		buffer[i] = rand() % 256;
	}
}

__device__ void getPayloadCuda(curandState_t* state, uint8_t *buffer, uint32_t idx)
{
	//First 0-80 bytes
	for (size_t i = 0; i < cPayloadSizeU32; i += cU32Size)
	{
		uint32_t u32RandL = curand(state);
		memcpy(&(buffer[i]), &u32RandL, cU32Size);
	}

	//Last 80-83 bytes
	uint32_t u32RandL = curand(state);
	memcpy(&(buffer[cPayloadSizeU32]), &u32RandL, (cPayloadSize - cPayloadSizeU32));
}

__global__ void gCUDA_CURAND_Init(uint8_t* u8TimeL)
{
	int idx = (blockDim.x*blockIdx.x) + threadIdx.x;
	if (idx < cNumberOfThreads)
	{
		uint32_t* u32TimeL = (uint32_t*)&u8TimeL;
		if (idx >= 1)
		{
			curand_init((unsigned long long)clock()*(*u32TimeL)*idx, 0, 0, &CurandStateHolder[idx]);
		}
		else
		{
			curand_init((unsigned long long)clock()*(*u32TimeL), 0, 0, &CurandStateHolder[idx]);
		}
	}
}

__global__ void gCUDA_Fill_Payload(payload_t* buffer)
{
	int idx = (blockDim.x*blockIdx.x) + threadIdx.x;
	if (idx < cNumberOfThreads)
	{
		getPayloadCuda(&CurandStateHolder[idx], buffer[idx].payload, idx);
	}
}

__global__ void gCUDA_Fill_Payload_Four_Byte(payload_t* buffer)
{
	int idx = (blockDim.x*blockIdx.x) + threadIdx.x;
	if (idx < cNumberOfThreads)
	{
		uint32_t u32RandL = curand(&CurandStateHolder[idx]);
		memcpy(&(buffer[idx].payload[cDICE_SHA3_512_SIZE]), &u32RandL, cU32Size);
	}
}