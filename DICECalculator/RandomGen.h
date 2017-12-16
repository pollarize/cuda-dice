#pragma once
class RandomGen
{
public:
	RandomGen();
	~RandomGen();

	void RandomGen::getPayload(uint8_t* buffer, uint8_t size);
};

