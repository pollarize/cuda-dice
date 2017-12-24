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

#include "SwatcBeats_C.h"
#include <time.h>

//Local const
const time_t _referentDateSeconds = 978303600; //"January 01, 2001 00:00:00 GMT+0100" in seconds
const float _beatsPerSecond = 0.011574;

//Static data
static time_t  TimeG;

SwatcBeats_C::SwatcBeats_C()
{
	time(&TimeG);
}

unsigned int SwatcBeats_C::getBeats()
{
	unsigned int iBeatsL = 0;

	//Read time
	time(&TimeG);

	//Calculate Beats by Ref date
	iBeatsL = (int)(TimeG - _referentDateSeconds);

	//Multiply to beats per seconds
	iBeatsL *= _beatsPerSecond;

	return iBeatsL;
}

SwatcBeats_C::~SwatcBeats_C()
{
	(void)TimeG;
}
