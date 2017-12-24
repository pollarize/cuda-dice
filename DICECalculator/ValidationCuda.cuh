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
#include <stdint.h>
#include <math.h>       /* floor */

#define cSizeOfDiceUnit ((uint16_t)1024/8) //in bytes
#define cU8_SIZE        ((uint8_t)8)
#define cSHA3_512_SIZE  ((uint16_t)512/8)  //in bytes
#define cZEROE          ((uint8_t)0)

__device__ __host__ void validateHash(uint16_t zeroes, uint8_t* hashProto, bool* isInvalid)
{
	uint8_t u8ChunksL = zeroes / cU8_SIZE;
	uint8_t u8PeiesOfChunksL = zeroes % cU8_SIZE;
	uint8_t u8LastByteValueL = 0b11111111 << u8PeiesOfChunksL;
	*isInvalid = !((u8LastByteValueL | hashProto[cSHA3_512_SIZE - u8ChunksL - 1]) == u8LastByteValueL);

	//If the last item is valid, 
	//check the previous are they 
	//all equal to zero
	if (false == *isInvalid) {
		for (size_t i = (cSHA3_512_SIZE - 1); i >= cSHA3_512_SIZE - u8ChunksL; i--) {
			if (cZEROE != hashProto[i]) {
				*isInvalid = true;
				break;
			}
		}
	}
	else {
		//Nothing To Do 
	}
}