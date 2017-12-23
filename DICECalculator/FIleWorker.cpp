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
