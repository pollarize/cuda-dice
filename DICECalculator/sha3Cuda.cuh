#pragma once
#include <string.h>
#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SHA3_512_SIZE  (uint8_t((512/8)))

#if defined(_MSC_VER)
#define SHA3_CONST(x) x
#else
#define SHA3_CONST(x) x##L
#endif

#ifndef SHA3_ROTL64
#define SHA3_ROTL64(x, y) \
	(((x) << (y)) | ((x) >> ((sizeof(uint64_t)*8) - (y))))
#endif

#define SHA3_ASSERT( x )
#if defined(_MSC_VER)
#define SHA3_TRACE( format, ...)
#define SHA3_TRACE_BUF( format, buf, l, ...)
#else
#define SHA3_TRACE(format, args...)
#define SHA3_TRACE_BUF(format, buf, l, args...)
#endif

/* 'Words' here refers to uint64_t */
#define SHA3_KECCAK_SPONGE_WORDS \
	(((1600)/8/*bits to byte*/)/sizeof(uint64_t))


__device__ __host__ void keccakf(uint64_t s[25])
{
	 const uint64_t keccakf_rndc[24] = {
		SHA3_CONST(0x0000000000000001UL), SHA3_CONST(0x0000000000008082UL),
		SHA3_CONST(0x800000000000808aUL), SHA3_CONST(0x8000000080008000UL),
		SHA3_CONST(0x000000000000808bUL), SHA3_CONST(0x0000000080000001UL),
		SHA3_CONST(0x8000000080008081UL), SHA3_CONST(0x8000000000008009UL),
		SHA3_CONST(0x000000000000008aUL), SHA3_CONST(0x0000000000000088UL),
		SHA3_CONST(0x0000000080008009UL), SHA3_CONST(0x000000008000000aUL),
		SHA3_CONST(0x000000008000808bUL), SHA3_CONST(0x800000000000008bUL),
		SHA3_CONST(0x8000000000008089UL), SHA3_CONST(0x8000000000008003UL),
		SHA3_CONST(0x8000000000008002UL), SHA3_CONST(0x8000000000000080UL),
		SHA3_CONST(0x000000000000800aUL), SHA3_CONST(0x800000008000000aUL),
		SHA3_CONST(0x8000000080008081UL), SHA3_CONST(0x8000000000008080UL),
		SHA3_CONST(0x0000000080000001UL), SHA3_CONST(0x8000000080008008UL)
	};

	 const unsigned keccakf_rotc[24] = {
		1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62,
		18, 39, 61, 20, 44
	};

	 const unsigned keccakf_piln[24] = {
		10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20,
		14, 22, 9, 6, 1
	};

#define KECCAK_ROUNDS 24

	int i, j, round;
	uint64_t t, bc[5];


	for (round = 0; round < KECCAK_ROUNDS; round++) {

		/* Theta */
		for (i = 0; i < 5; i++)
			bc[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];

		for (i = 0; i < 5; i++) {
			t = bc[(i + 4) % 5] ^ SHA3_ROTL64(bc[(i + 1) % 5], 1);
			for (j = 0; j < 25; j += 5)
				s[j + i] ^= t;
		}

		/* Rho Pi */
		t = s[1];
		for (i = 0; i < 24; i++) {
			j = keccakf_piln[i];
			bc[0] = s[j];
			s[j] = SHA3_ROTL64(t, keccakf_rotc[i]);
			t = bc[0];
		}

		/* Chi */
		for (j = 0; j < 25; j += 5) {
			for (i = 0; i < 5; i++)
				bc[i] = s[j + i];
			for (i = 0; i < 5; i++)
				s[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
		}

		/* Iota */
		s[0] ^= keccakf_rndc[round];
	}
}

/* *************************** Public Inteface ************************ */
__device__ __host__ void sha3_SingleExeuction(void const *bufIn, size_t len, uint8_t* bufOut)
{
	struct  sha3_context {
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
	};

	//Init SHA3-512
	sha3_context constexL;
	memset(&constexL, 0, sizeof(constexL));
	constexL.capacityWords = 2 * 512 / (8 * sizeof(uint64_t));

	/* 0...7 -- how much is needed to have a word */
	unsigned old_tail = (8 - constexL.byteIndex) & 7;

	size_t words;
	unsigned tail;
	size_t i;

	const uint8_t *buf = (const uint8_t*)bufIn;

	SHA3_TRACE_BUF("called to update with:", buf, len);

	SHA3_ASSERT(constexL.byteIndex < 8);
	SHA3_ASSERT(constexL.wordIndex < sizeof(constexL.s) / sizeof(constexL.s[0]));

	if (len < old_tail) {        /* have no complete word or haven't started
								 * the word yet */
		SHA3_TRACE("because %d<%d, store it and return", (unsigned)len,
			(unsigned)old_tail);
		/* endian-independent code follows: */
		while (len--)
			constexL.saved |= (uint64_t)(*(buf++)) << ((constexL.byteIndex++) * 8);
		SHA3_ASSERT(constexL.byteIndex < 8);
	}

	if (old_tail) {              /* will have one word to process */
		SHA3_TRACE("completing one word with %d bytes", (unsigned)old_tail);
		/* endian-independent code follows: */
		len -= old_tail;
		while (old_tail--)
			constexL.saved |= (uint64_t)(*(buf++)) << ((constexL.byteIndex++) * 8);

		/* now ready to add saved to the sponge */
		constexL.s[constexL.wordIndex] ^= constexL.saved;
		SHA3_ASSERT(constexL.byteIndex == 8);
		constexL.byteIndex = 0;
		constexL.saved = 0;
		if (++constexL.wordIndex ==
			(SHA3_KECCAK_SPONGE_WORDS - constexL.capacityWords)) {
			keccakf(constexL.s);
			constexL.wordIndex = 0;
		}
	}

	/* now work in full words directly from input */

	SHA3_ASSERT(constexL.byteIndex == 0);

	words = len / sizeof(uint64_t);
	tail = len - words * sizeof(uint64_t);

	SHA3_TRACE("have %d full words to process", (unsigned)words);

	for (i = 0; i < words; i++, buf += sizeof(uint64_t)) {
		const uint64_t t = (uint64_t)(buf[0]) |
			((uint64_t)(buf[1]) << 8 * 1) |
			((uint64_t)(buf[2]) << 8 * 2) |
			((uint64_t)(buf[3]) << 8 * 3) |
			((uint64_t)(buf[4]) << 8 * 4) |
			((uint64_t)(buf[5]) << 8 * 5) |
			((uint64_t)(buf[6]) << 8 * 6) |
			((uint64_t)(buf[7]) << 8 * 7);
#if defined(__x86_64__ ) || defined(__i386__)
		SHA3_ASSERT(memcmp(&t, buf, 8) == 0);
#endif
		constexL.s[constexL.wordIndex] ^= t;
		if (++constexL.wordIndex ==
			(SHA3_KECCAK_SPONGE_WORDS - constexL.capacityWords)) {
			keccakf(constexL.s);
			constexL.wordIndex = 0;
		}
	}

	SHA3_TRACE("have %d bytes left to process, save them", (unsigned)tail);

	/* finally, save the partial word */
	SHA3_ASSERT(constexL.byteIndex == 0 && tail < 8);
	while (tail--) {
		SHA3_TRACE("Store byte %02x '%c'", *buf, *buf);
		constexL.saved |= (uint64_t)(*(buf++)) << ((constexL.byteIndex++) * 8);
	}
	SHA3_ASSERT(constexL.byteIndex < 8);
	SHA3_TRACE("Have saved=0x%016" PRIx64 " at the end", constexL.saved);

	//Finalize
	SHA3_TRACE("called with %d bytes in the buffer", constexL.byteIndex);

	/* Append 2-bit suffix 01, per SHA-3 spec. Instead of 1 for padding we
	* use 1<<2 below. The 0x02 below corresponds to the suffix 01.
	* Overall, we feed 0, then 1, and finally 1 to start padding. Without
	* M || 01, we would simply use 1 to start padding. */

#ifndef SHA3_USE_KECCAK
	/* SHA3 version */
	constexL.s[constexL.wordIndex] ^=
		(constexL.saved ^ ((uint64_t)((uint64_t)(0x02 | (1 << 2)) <<
		((constexL.byteIndex) * 8))));
#else
	/* For testing the "pure" Keccak version */
	constexL.s[constexL.wordIndex] ^=
		(constexL.saved ^ ((uint64_t)((uint64_t)1 << (constexL.byteIndex *
			8))));
#endif

	constexL.s[SHA3_KECCAK_SPONGE_WORDS - constexL.capacityWords - 1] ^=
		SHA3_CONST(0x8000000000000000UL);
	keccakf(constexL.s);

	/* Return first bytes of the constexL.s. This conversion is not needed for
	* little-endian platforms e.g. wrap with #if !defined(__BYTE_ORDER__)
	* || !defined(__ORDER_LITTLE_ENDIAN__) || \
	* __BYTE_ORDER__!=__ORDER_LITTLE_ENDIAN__ ... the conversion below ...
	* #endif */
	{
		unsigned i;
		for (i = 0; i < SHA3_KECCAK_SPONGE_WORDS; i++) {
			const unsigned t1 = (uint32_t)constexL.s[i];
			const unsigned t2 = (uint32_t)((constexL.s[i] >> 16) >> 16);
			constexL.sb[i * 8 + 0] = (uint8_t)(t1);
			constexL.sb[i * 8 + 1] = (uint8_t)(t1 >> 8);
			constexL.sb[i * 8 + 2] = (uint8_t)(t1 >> 16);
			constexL.sb[i * 8 + 3] = (uint8_t)(t1 >> 24);
			constexL.sb[i * 8 + 4] = (uint8_t)(t2);
			constexL.sb[i * 8 + 5] = (uint8_t)(t2 >> 8);
			constexL.sb[i * 8 + 6] = (uint8_t)(t2 >> 16);
			constexL.sb[i * 8 + 7] = (uint8_t)(t2 >> 24);
		}
	}

	SHA3_TRACE_BUF("Hash: (first 32 bytes)", constexL.sb, 256 / 8);

	//Copy data from local buffer
	memcpy(bufOut, constexL.sb, SHA3_512_SIZE);
}