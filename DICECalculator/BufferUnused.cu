//
////testHashAndValidation();
//testDiceCalc(0);
//{
//	//std::thread first(testDiceCalc, 0);     // spawn new thread that calls foo()
//	////std::thread second(testDiceCalc, 1);  // spawn new thread that calls bar(0)
//
//	////std::thread third(testDiceCalc, 2);     // spawn new thread that calls foo()
//	////std::thread fourth(testDiceCalc, 3);  // spawn new thread that calls bar(0)
//
//	////std::thread fifth(testDiceCalc, 4);     // spawn new thread that calls foo()
//	////std::thread sixth(testDiceCalc, 5);  // spawn new thread that calls bar(0)
//
//	//std::cout << "Six threads started...\n";
//
//
//	//// synchronize threads:
//	//first.join();
//	//second.join();
//
//	//third.join();
//	//fourth.join();
//
//	//fifth.join();
//	//sixth.join();
//}
////memset(c, 0, size);
////char* defaultString = "Hello World";
////memcpy(c->shaPayload, defaultString, strlen(defaultString));
//
////auto startTimer = chrono::steady_clock::now();
////   cudaError_t cudaStatus = hashWithCuda(c, size);
////auto endTimer = chrono::steady_clock::now();
//
////wcout << endl << c << endl;
//
////cout << "Threads per second : "
////	<< (cArraySize/chrono::duration_cast<chrono::milliseconds>(endTimer - startTimer).count())*1000
////	<<  endl;
//
////cout << "Elapsed time in milliseconds : "
////	<< chrono::duration_cast<chrono::milliseconds>(endTimer - startTimer).count()
////	<< " ms" << endl;
//
////startTimer = chrono::steady_clock::now();
////////for (size_t i = 0; i < props.maxThreadsDim[0]* props.maxThreadsDim[1]; i++)
////////{
////////	uint8_t bufferL[SHA3_512_SIZE];
////////	sha3_SingleExeuction("Hello World", 12, bufferL);
////////}
//////for (int i = 0; i < cArraySize; i++)
//////{
//////	getPayload(diceUnits[i].payload, cPayloadSize);
//////}
//////cudaStatus = randWithCuda(diceUnits, size);
//
////endTimer = chrono::steady_clock::now();
//
////cout << "Elapsed time in nanoseconds : "
////	<< chrono::duration_cast<chrono::milliseconds>(endTimer - startTimer).count()
////	<< " ms" << endl;
//
////   if (cudaStatus != cudaSuccess) {
////       fprintf(stderr, "addWithCuda failed!");
////       return 1;
////   }
//
////   // cudaDeviceReset must be called before exiting in order for profiling and
////   // tracing tools such as Nsight and Visual Profiler to show complete traces.
////   cudaStatus = cudaDeviceReset();
////   if (cudaStatus != cudaSuccess) {
////       fprintf(stderr, "cudaDeviceReset failed!");
////       return 1;
////   }