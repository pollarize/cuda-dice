#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdint.h>
#include <string.h>

__device__ __host__  void getBeats(uint8_t* buffer)
{
	uint32_t u32BeatsL = 0;
	static time_t  TimeG;

	//Local const
	const time_t _referentDateSeconds = 978303600; //"January 01, 2001 00:00:00 GMT+0100" in seconds
	const float _beatsPerSecond = 0.011574;

	//Read time
	time(&TimeG);

	//Calculate Beats by Ref date
	u32BeatsL = (uint32_t)(TimeG - _referentDateSeconds);

	//Multiply to beats per seconds
	u32BeatsL *= _beatsPerSecond;

	//Save data
	memcpy(buffer, &u32BeatsL, sizeof(uint32_t));
}
