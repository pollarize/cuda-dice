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
