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