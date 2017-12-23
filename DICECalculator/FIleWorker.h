#pragma once
class FIleWorker
{
public:
	FIleWorker::FIleWorker();
	~FIleWorker();

	int FIleWorker::writeToFile(char* outputFilePath, diceUnitHex_t* diceUnitP);
};

