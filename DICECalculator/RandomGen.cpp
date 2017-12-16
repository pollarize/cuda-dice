#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include "RandomGen.h"

#define cPayloadSize ((uint8_t)64)

RandomGen::RandomGen()
{
}


RandomGen::~RandomGen()
{
}

void RandomGen::getPayload(uint8_t* buffer, uint8_t size)
{
	srand(time(NULL));

	for (size_t i = 0; i < size; i++)
	{
		buffer[i] = rand() % 256;
	}
}
