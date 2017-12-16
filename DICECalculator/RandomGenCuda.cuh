#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <curand_kernel.h>

#define cPayloadSize ((uint8_t)83)

__host__ void getPayload(uint8_t* buffer, int size)
{
	srand(time(NULL));

	for (size_t i = 0; i < size; i++)
	{
		buffer[i] = rand() % 256;
	}
}

__device__ void getPayloadCuda(uint8_t *buffer, int size)
{
	curandState state;
	curand_init((unsigned long long)clock(), 0, 0, &state);
	for (size_t i = 0; i < size; i++)
	{
		buffer[i] = (uint8_t)curand_uniform_double(&state);
	}
}