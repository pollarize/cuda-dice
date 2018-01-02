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
#include <time.h>
#include <stdint.h>
#include <string.h>

__host__  void getBeats(uint8_t* buffer)
{
	uint32_t u32BeatsL = 0;
	time_t  TimeG;
	uint8_t *u32BeatsLp;

	//Local const
	const time_t _referentDateSeconds = 978303600; //"January 01, 2001 00:00:00 GMT+0100" in seconds
	const double _beatsPerSecond = 0.011574;

	//Read time
	time(&TimeG);

	//Calculate Beats by Ref date
	u32BeatsL = (uint32_t)(TimeG - _referentDateSeconds);

	//Multiply to beats per seconds
	u32BeatsL = (uint32_t)(u32BeatsL *_beatsPerSecond);

	//Reverse order (machine code related)
	u32BeatsLp = (uint8_t *)&u32BeatsL;
	buffer[0] = u32BeatsLp[3];
	buffer[1] = u32BeatsLp[2];
	buffer[2] = u32BeatsLp[1];
	buffer[3] = u32BeatsLp[0];
}
