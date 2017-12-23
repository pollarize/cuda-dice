#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdint.h>
#include <string.h>

__device__ __host__  void getBeats(uint8_t* buffer)
{
	uint32_t u32BeatsL = 0;
    time_t  TimeG;
	uint8_t *u32BeatsLp;

	//Local const
	const time_t _referentDateSeconds = 978303600; //"January 01, 2001 00:00:00 GMT+0100" in seconds
	const float _beatsPerSecond = 0.011574;

	//Read time
	time(&TimeG);

	//Calculate Beats by Ref date
	u32BeatsL = (uint32_t)(TimeG - _referentDateSeconds);

	//Multiply to beats per seconds
	u32BeatsL *= _beatsPerSecond;

	//Reverse order (machine code related)
	u32BeatsLp = (uint8_t *)&u32BeatsL;
	buffer[0] = u32BeatsLp[3];
	buffer[1] = u32BeatsLp[2];
	buffer[2] = u32BeatsLp[1];
	buffer[3] = u32BeatsLp[0];
}
