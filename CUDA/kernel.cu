
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <iostream>
#include "kernel.h"

#ifdef __CUDACC__
	#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
	#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
	#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
	#define KERNEL_ARGS2(grid, block)
	#define KERNEL_ARGS3(grid, block, sh_mem)
	#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

using namespace std;

#define NUM_BITS ((unsigned int)(8 * sizeof(unsigned int)))

const unsigned int data_length = 256;

const unsigned int THREADS = 256;
const unsigned int FILTER_LENGTH = 3;
const unsigned int NUM_FILTERS = 2;
const unsigned int FILTERS = (0b111 << 3) | (0b011);

char nums[2] = { '0', '1' };

#define dataArrayLength ((data_length + NUM_BITS - 1) / NUM_BITS)
#define resultArrayLength ((data_length * NUM_FILTERS + NUM_BITS - 1) / NUM_BITS)



__global__
void convEncode(const unsigned int *data, unsigned int *output) {
	unsigned int start = blockIdx.x * (THREADS / NUM_BITS);
	unsigned int idx = threadIdx.x;
	//printf("%d\n", idx);

	unsigned int val = (((((idx / NUM_BITS) | start) != 0 
			? data[idx / NUM_BITS + start - 1] << (1 + (idx % NUM_BITS)) 
			: 0))
		| (data[idx / NUM_BITS + start] >> (NUM_BITS - 1 - (idx % NUM_BITS))))
		& ((1 << FILTER_LENGTH) - 1);
	
	/*printf("%d\t%d\t%x\n", idx / NUM_BITS + start - 1, idx / NUM_BITS + start, (((((idx / NUM_BITS) | start) != 0
		? data[idx / NUM_BITS + start - 1] << (1 + (idx % NUM_BITS))
		: 0))
		| (data[idx / NUM_BITS + start] >> (NUM_BITS - 1 - (idx % NUM_BITS))))
		& ((1 << FILTER_LENGTH) - 1));*/

	for (unsigned int i = 0; i < NUM_FILTERS; i++) {

		unsigned int n = val & ((FILTERS >> ((NUM_FILTERS - i - 1) * FILTER_LENGTH))
			& ((1 << FILTER_LENGTH) - 1));

		n = n - ((n >> 1) & 0x55555555);
		n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
		unsigned int newVal = (((((n + (n >> 4) & 0x0F0F0F0F) * 0x01010101) >> 24) & 1));

		output[NUM_FILTERS * start + (NUM_FILTERS * idx) / NUM_BITS] |=
			((unsigned int)(newVal << ((NUM_BITS * NUM_FILTERS - 1 - (NUM_FILTERS * (idx % NUM_BITS) + i)))));

		//printf("%x\n", (newVal << ((NUM_BITS * NUM_FILTERS - 1 - (NUM_FILTERS * (idx % NUM_BITS) + i)))));
	}


	/*unsigned int val = ((
		(idx & start 
			? data[idx / 8 + start - 1] << 8 : 0) 
		| data[idx / 8 + start]) >> (7 - (idx % 8))) 
		& (~(1 << FILTER_LENGTH));

	for (int i = 0; i < NUM_FILTERS; i++) {
		
		unsigned int n = val ^ ((FILTERS >> (i * FILTER_LENGTH)) & ~(1 << FILTER_LENGTH));
		
		n = n - ((n >> 1) & 0x55);
		n = (n & 0x33) + ((n >> 2) & 0x33);

		output[NUM_FILTERS * start + (NUM_FILTERS * idx) / 4] |= ((
			((n + (n >> 4) & 0xF) * 0x01010101) >> 24) & 1)
			<< (16 - (2 * (idx % 8) + i));
	}*/
}

void printBitString(const unsigned int *data, int length) {
	//ostringstream strStream;
	for (int i = 0; i < length; i++) {
		for (int j = 0; j < NUM_BITS; j++) {
			cout << nums[(data[i] >> (NUM_BITS - j)) & 1];
		}
	}
	cout << endl;
}

int main() {
	unsigned int *A, *B;

	printf("%x\n", ((1 << FILTER_LENGTH) - 1));
	
	cudaMallocManaged(&A, dataArrayLength * sizeof(unsigned int));
	cudaMallocManaged(&B, resultArrayLength * sizeof(unsigned int));

	for (unsigned int i = 0; i < dataArrayLength; i ++) {
		A[i] = 0xFFFFFFFF;
	}

	for (unsigned int i = 0; i < resultArrayLength; i++) {
		B[i] = 0;
	}

	cout << resultArrayLength << endl;

	printBitString(A, dataArrayLength);
	printBitString(B, resultArrayLength);

	// This macro is used because the compiler throws a compile error when trying to parse 
	// the correct CUDA kernel syntax
	convEncode KERNEL_ARGS2((data_length + THREADS - 1) / THREADS, THREADS) (A, B);

	cudaDeviceSynchronize();

	int errors = 0;

	for (int i = 0; i < resultArrayLength; i ++) {
		printf("%x", B[i]);
	}

	unsigned int expected[] = { 0xca, 0x60000000 };

	for (int i = 0; i < resultArrayLength; i++) {
		errors += fmin((B[i]) ^ expected[i % 2], 1);
	}

	std::cout << "\nTotal Error: " << errors << std::endl;

	printBitString(B, resultArrayLength);

	printBitString(A, dataArrayLength);

	cudaFree(A);
	cudaFree(B);

	return 0;
}


