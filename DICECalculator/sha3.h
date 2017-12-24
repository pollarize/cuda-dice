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

//Local macroses
#define cByteSize       ((uint8_t)8)
#define cSHASize        ((uint16_t)512)
#define cStringArrayMax ((uint16_t)1024)
#define OPTIMAZE        ((uint8_t)1)

/* 'Words' here refers to uint64_t */
#define SHA3_KECCAK_SPONGE_WORDS \
	(((1600)/8/*bits to byte*/)/sizeof(uint64_t))

//Local function Prototypes
extern const uint8_t* CalculateSHA3(const char * pU8ArrayP, uint8_t u8SizeP);
extern uint8_t* StringToHex(uint8_t* pCharArrayP, uint8_t u8CountOfBytesP);

typedef struct sha3_context_ {
	uint64_t saved;             /* the portion of the input message that we
								* didn't consume yet */
	union {                     /* Keccak's state */
		uint64_t s[SHA3_KECCAK_SPONGE_WORDS];
		uint8_t sb[SHA3_KECCAK_SPONGE_WORDS * 8];
	};
	unsigned byteIndex;         /* 0..7--the next byte after the set one
								* (starts from 0; 0--none are buffered) */
	unsigned wordIndex;         /* 0..24--the next word to integrate input
								* (starts from 0) */
	unsigned capacityWords;     /* the double size of the hash output in
								* words (e.g. 16 for Keccak 512) */
} sha3_context;