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