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


#include <iostream>
#include <fstream>
#include "DiceCalcCudaTypes.cuh"
#include "FIleWorker.h"

using namespace std;

FIleWorker::FIleWorker()
{
}

int FIleWorker::writeToFile(char* outputFilePath, diceUnitHex_t* diceUnitP)
{
	char buffer[1024];
	int n;
	std::ofstream outfile(outputFilePath, std::ofstream::binary);
	n = sprintf(buffer, "\{\"addrOperator\": \"%s\",\"addrMiner\" : \"%s\",\"validZeros\" : \"%s\",\"swatchTime\" : \"%s\",	\"payLoad\" : \"%s\" \}",diceUnitP->addrOp, diceUnitP->addrMin, diceUnitP->validZeroes, diceUnitP->swatchTime, diceUnitP->payload);
	outfile.write(buffer, n);
	outfile.close();
	return 0;
}



FIleWorker::~FIleWorker()
{
}
