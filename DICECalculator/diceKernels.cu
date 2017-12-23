///*User libraries for CUDA*/
//#include "sha3Cuda.cuh"
//#include "SwatchCuda.cuh"
//#include "RandomGenCuda.cuh"
//#include "ValidationCuda.cuh"
//#include "DiceCalcCuda.cuh"
//
////Function Executed on GPU
//__global__ void  calculateDICE(hasinig_t *c)
//{
//	int i = (blockDim.x*blockIdx.x) + threadIdx.x;
//
//	sha3_SingleExeuction("Hello World", 12, c[i].buffer);
//
//	c[i].isValid = true;
//}
//
//__global__ void cudaRandPayload(payload_t *d_out)
//{
//	int i = (blockDim.x*blockIdx.x) + threadIdx.x;
//	getPayloadCuda(d_out[i].payload, cPayloadSize);
//}