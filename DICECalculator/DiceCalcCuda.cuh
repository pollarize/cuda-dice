#pragma once
#include <stdint.h>

typedef struct diceUnit
{
	uint8_t addrOp[20];
	uint8_t addrMin[20];
	uint8_t validZeroes[1];
	uint8_t swatchTime[4];
	uint8_t payload[83];
}diceUnit_t;

typedef struct diceHeader
{
	uint8_t addrOp[20];
	uint8_t addrMin[20];
	uint8_t validZeroes[1];
}diceHeader_t;

typedef struct diceProto
{
	uint8_t addrOp[20];
	uint8_t addrMin[20];
	uint8_t validZeroes[1];
	uint8_t swatchTime[4];
	uint8_t shaPayload[64];
}diceProto_t;

typedef struct diceHash
{
	uint8_t payload[83];
	uint8_t shaPayload[64];
}diceHash_t;

typedef struct Payload
{
	uint8_t payload[83];
}payload_t;

typedef struct hashPayload
{
	uint8_t shaPayload[64];
}hashPayload_t;