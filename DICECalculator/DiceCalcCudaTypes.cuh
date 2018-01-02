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
#include <stdint.h>

/*All of sizes are in bytes*/
#ifndef DICE_TYPE_SIZES 
#define cDICE_ADDR_SIZE             ((uint8_t)20)
#define cDICE_ZEROES_SIZE           ((uint8_t)1)
#define cDICE_SWATCH_TIME_SIZE      ((uint8_t)4)
#define cDICE_PAYLOAD_SIZE          ((uint8_t)83)
#define cDICE_SHA3_512_SIZE         ((uint8_t)64)
#define cDICE_UNIT_SIZE             (cDICE_ADDR_SIZE+cDICE_ADDR_SIZE+cDICE_ZEROES_SIZE+cDICE_SWATCH_TIME_SIZE+cDICE_PAYLOAD_SIZE)
#define cDICE_PROTO_SIZE            (cDICE_ADDR_SIZE+cDICE_ADDR_SIZE+cDICE_ZEROES_SIZE+cDICE_SWATCH_TIME_SIZE+cDICE_SHA3_512_SIZE)

#define cBYTE_TO_HEX                ((uint8_t)2)
#endif

#ifdef DICE_BYTE

//Remove HEX size
#undef  cBYTE_TO_HEX
#define cBYTE_TO_HEX ((uint8_t)1) 

#endif // DICE_BYTE

typedef struct diceUnitHex
{
	uint8_t addrOp[cDICE_ADDR_SIZE *cBYTE_TO_HEX];
	uint8_t addrMin[cDICE_ADDR_SIZE *cBYTE_TO_HEX];
	uint8_t validZeroes[cDICE_ZEROES_SIZE *cBYTE_TO_HEX];
	uint8_t swatchTime[cDICE_SWATCH_TIME_SIZE *cBYTE_TO_HEX];
	uint8_t payload[cDICE_PAYLOAD_SIZE *cBYTE_TO_HEX];
}diceUnit_t;


typedef struct diceProtoHex
{
	uint8_t addrOp[cDICE_ADDR_SIZE *cBYTE_TO_HEX];
	uint8_t addrMin[cDICE_ADDR_SIZE *cBYTE_TO_HEX];
	uint8_t validZeroes[cDICE_ZEROES_SIZE *cBYTE_TO_HEX];
	uint8_t swatchTime[cDICE_SWATCH_TIME_SIZE *cBYTE_TO_HEX];
	uint8_t shaPayload[cDICE_SHA3_512_SIZE *cBYTE_TO_HEX];
}diceProto_t;

typedef struct Payload
{
	uint8_t payload[cDICE_PAYLOAD_SIZE];
}payload_t;


typedef struct hashProtoHex
{
	uint8_t hashProto[cDICE_SHA3_512_SIZE *cBYTE_TO_HEX];
}hashProto_t;

