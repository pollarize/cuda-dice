#pragma once
#include <stdint.h>

/*All of sizes are in bytes*/
#ifndef DICE_TYPE_SIZES 
#define cDICE_ADDR_SIZE             ((uint8_t)20)
#define cDICE_ZEROES_SIZE           ((uint8_t)1)
#define cDICE_SWATCH_TIME_SIZE      ((uint8_t)4)
#define cDICE_PAYLOAD_SIZE          ((uint8_t)83)
#define cDICE_SHA3_512_SIZE         ((uint8_t)64)

#define cBYTE_TO_HEX                ((uint8_t)2)
#endif


typedef struct diceUnit
{
	uint8_t addrOp[cDICE_ADDR_SIZE];
	uint8_t addrMin[cDICE_ADDR_SIZE];
	uint8_t validZeroes[cDICE_ZEROES_SIZE];
	uint8_t swatchTime[cDICE_SWATCH_TIME_SIZE];
	uint8_t payload[cDICE_PAYLOAD_SIZE];
}diceUnit_t;

typedef struct diceUnitHex
{
	uint8_t addrOp[cDICE_ADDR_SIZE *cBYTE_TO_HEX];
	uint8_t addrMin[cDICE_ADDR_SIZE *cBYTE_TO_HEX];
	uint8_t validZeroes[cDICE_ZEROES_SIZE *cBYTE_TO_HEX];
	uint8_t swatchTime[cDICE_SWATCH_TIME_SIZE *cBYTE_TO_HEX];
	uint8_t payload[cDICE_PAYLOAD_SIZE *cBYTE_TO_HEX];
}diceUnitHex_t;

typedef struct diceHeader
{
	uint8_t addrOp[cDICE_ADDR_SIZE];
	uint8_t addrMin[cDICE_ADDR_SIZE];
	uint8_t validZeroes[cDICE_ZEROES_SIZE];
}diceHeader_t;

typedef struct diceProto
{
	uint8_t addrOp[cDICE_ADDR_SIZE];
	uint8_t addrMin[cDICE_ADDR_SIZE];
	uint8_t validZeroes[cDICE_ZEROES_SIZE];
	uint8_t swatchTime[cDICE_SWATCH_TIME_SIZE];
	uint8_t shaPayload[cDICE_SHA3_512_SIZE];
}diceProto_t;

typedef struct diceProtoHex
{
	uint8_t addrOp[cDICE_ADDR_SIZE *cBYTE_TO_HEX];
	uint8_t addrMin[cDICE_ADDR_SIZE *cBYTE_TO_HEX];
	uint8_t validZeroes[cDICE_ZEROES_SIZE *cBYTE_TO_HEX];
	uint8_t swatchTime[cDICE_SWATCH_TIME_SIZE *cBYTE_TO_HEX];
	uint8_t shaPayload[cDICE_SHA3_512_SIZE *cBYTE_TO_HEX];
}diceProtoHEX_t;

typedef struct cudaHashPayload
{
	uint8_t payload[cDICE_PAYLOAD_SIZE];
	uint8_t shaPayload[cDICE_SHA3_512_SIZE];
}cudaHashPayload_t;

typedef struct Payload
{
	uint8_t payload[cDICE_PAYLOAD_SIZE];
}payload_t;

typedef struct hashPayload
{
	uint8_t shaPayload[cDICE_SHA3_512_SIZE];
}hashPayload_t;

typedef struct hashProtoHex
{
	uint8_t hashProto[cDICE_SHA3_512_SIZE *cBYTE_TO_HEX];
}hashProtoHex_t;

