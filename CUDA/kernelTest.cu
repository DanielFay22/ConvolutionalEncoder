
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <chrono>

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

// Some commonly referenced values
#define NUM_BITS ((unsigned int)(8 * sizeof(unsigned int)))
#define dataArrayLength ((data_length + NUM_BITS - 1) / NUM_BITS)
#define resultArrayLength ((data_length * NUM_FILTERS + NUM_BITS - 1) / NUM_BITS)

// Macro for automatically checking and reporting CUDA errors
#define CUDACHECK(cmd) do { \
    cudaError_t e = cmd;    \
    if( e != cudaSuccess ) { \
    printf("Failed: Cuda error %s:%d '%s'\n", \
        __FILE__,__LINE__,cudaGetErrorString(e)); \
    exit(EXIT_FAILURE);     \
  } \
} while(0)

// Constants defined for encoding
unsigned int data_length = 256;
const unsigned int THREADS = 256;
const unsigned int FILTER_LENGTH = 3;
const unsigned int NUM_FILTERS = 2;
const unsigned int FILTERS = (0b111 << 3) | (0b011);

// misc global data
const int test_iters = 1;
char nums[2] = { '0', '1' };


__global__
void convEncode(const unsigned int *data, unsigned int *output) {
	unsigned int start = blockIdx.x * (THREADS / NUM_BITS);
	unsigned int idx = threadIdx.x;

	// Retrieve the relevant bits from the data and and shift them to LSB
	unsigned int val = (((((idx / NUM_BITS) | start) == 0 ? 0					// Don't attempt to look up non-existant values
			: data[idx / NUM_BITS + start - 1] << (1 + (idx % NUM_BITS))))		// Retrieve both the previous value in the array
		| (data[idx / NUM_BITS + start] >> (NUM_BITS - 1 - (idx % NUM_BITS))))	// and the current, in case the filter lies on a word boundary
		& ((1 << FILTER_LENGTH) - 1);
	
	__shared__ char bits[THREADS];

	bits[idx] = 0;

	__syncthreads();

#pragma unroll
	// Each thread evaluates all of the filters on its data
	for (unsigned int i = 0; i < NUM_FILTERS; i++) {

		unsigned int n = val & ((FILTERS >> ((NUM_FILTERS - i - 1) * FILTER_LENGTH))
			& ((1 << FILTER_LENGTH) - 1));

		n ^= n >> 1;                            // Parity of pairs of bits
		n ^= n >> 2;                            // Parity of every 4 bits
		n = (n & 0x11111111U) * 0x11111111U;    // Multiply to combine all the 4 bit parities
		n = (n >> 28) & 1;						// Shift and mask the parity bit to the LSB

		bits[idx] |= (n << (NUM_FILTERS - 1 - i));
	}

	__syncthreads();

	// Write the result from the shared memory to device memory. 
	// Every element in the output array is handled by a separate thread
	// to prevent errors from threads overwriting each other.
	if (idx < NUM_FILTERS * THREADS / NUM_BITS) {
		unsigned int o = 0;
		for (int i = idx * NUM_BITS / NUM_FILTERS; i < (idx + 1) * NUM_BITS / NUM_FILTERS; i++) {
			o <<= NUM_FILTERS;
			o |= bits[i] & (~(1 << NUM_FILTERS));
		}
		output[NUM_FILTERS * start + idx] = o;
	}
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

int hex2int(char ch) {
	if (ch >= '0' && ch <= '9')
		return ch - '0';
	if (ch >= 'A' && ch <= 'F')
		return ch - 'A' + 10;
	if (ch >= 'a' && ch <= 'f')
		return ch - 'a' + 10;
	return -1;
}

unsigned int* loadInpData(const char *inp) {
	int messageLen = strlen(inp);
	printf("Message length: %d", messageLen);
	unsigned int *data = (unsigned int*)(malloc(messageLen));
	for (int i = 0; i < messageLen; i++) {
		data[i / (sizeof(unsigned int) / sizeof(char))] <<= 8;
		data[i / (sizeof(unsigned int) / sizeof(char))] |= (0xFF & (hex2int(inp[i])));
	}

	return data;
}

long long runTest(unsigned long length, unsigned int fill, unsigned int numIters) {
	unsigned int *A, *B;

	CUDACHECK(cudaMallocManaged(&A, ((length + NUM_BITS - 1) / NUM_BITS) * sizeof(unsigned int)));
	CUDACHECK(cudaMallocManaged(&B, ((length * NUM_FILTERS + NUM_BITS - 1) / NUM_BITS) * sizeof(unsigned int)));

	CUDACHECK(cudaMemset(A, fill, ((length + NUM_BITS - 1) / NUM_BITS) * sizeof(unsigned int)));
	CUDACHECK(cudaMemset(B, 0, ((length * NUM_FILTERS + NUM_BITS - 1) / NUM_BITS) * sizeof(unsigned int)));

	auto start = chrono::high_resolution_clock::now();

	for (unsigned int j = 0; j < numIters; j++) {
		// This macro is used because the compiler throws a compile error when trying to parse 
		// the correct CUDA kernel syntax
		convEncode KERNEL_ARGS2((length + THREADS - 1) / THREADS, THREADS) (A, B);

		CUDACHECK(cudaDeviceSynchronize());
	}

	auto end = chrono::high_resolution_clock::now();

	long long elapsed_time = chrono::duration_cast<chrono::microseconds>(end - start).count();

	cudaFree(A);
	cudaFree(B);

	return elapsed_time;
}

/**
 * When called from the command line, the following arguments can be passed:
 * -t -- Flag to indicate the code should execute a performance test. Any arguments following this flag will be ignored
 * -m -- Use user provided data. Data should immediately follow the flag and should be in hex format.
 * -l -- Specifies a custom message length, in bits. If the -m flag is also used, this will have no effect.
 * -f -- Specifies the 32 bit value to fill the input array with. All formats accepted, with appropriate prefix (0b, 0x, etc.)
 * 		 If -m flag is set, has no effect.
 */
int main(int argc, char *argv[]) {
	unsigned int fillVal = 0xFFFFFFFF;
	bool useProvidedData = false, runtest = false;
	unsigned int *data;

	// Parse command line arguments
	// Usage:
	if (argc > 1) {
		for (int i = 1; i < argc; i++) {
			if (argv[i][0] == '-') {
				//printf("%c\t%d\t%d\n", argv[i][1], i, argv[i][1] == 'l');
				switch (argv[i][1]) {
				case 'L':
				case 'l': 
					i++; 
					if (i < argc and !useProvidedData) { data_length = (NUM_BITS - 1 + (unsigned int)(strtol(argv[i], NULL, 0))) / NUM_BITS * NUM_BITS; }
					break;
				case 'F':
				case 'f':
					i++;
					if (i < argc) { fillVal = (unsigned int)(strtol(argv[i], NULL, 0)); }
					break;
				case 'M':
				case 'm':
					i++;
					if (i < argc) {
						data = loadInpData(argv[i]);
						useProvidedData = true;
					}
					break;
				case 't':
				case 'T':
					i = argc;
					useProvidedData = false, runtest = true;
					break;
				default:
					printf("Invalid argument: %s", argv[i]);
					exit(1);
				}
			}
		}
	}

	unsigned int *A, *B;

	CUDACHECK(cudaMallocManaged(&A, dataArrayLength * sizeof(unsigned int)));
	CUDACHECK(cudaMallocManaged(&B, resultArrayLength * sizeof(unsigned int)));

	if (!useProvidedData) {
		memset(A, fillVal, dataArrayLength * sizeof(unsigned int));
	}
	else {
		memcpy(A, data, sizeof(unsigned int) * dataArrayLength);
	}
	CUDACHECK(cudaMemset(B, 0, resultArrayLength * sizeof(unsigned int)));

	// Run the kernel once to reduce timing errors from initial run
	convEncode KERNEL_ARGS2((data_length + THREADS - 1) / THREADS, THREADS) (A, B);
	CUDACHECK(cudaDeviceSynchronize());

	if (runtest) {
		long long times[31];
		for (int j = 0; j < test_iters; j++) {
			for (int i = 0; i < 31; i++) {
				if (j == 0) { times[i] = 0; }
				times[i] += runTest(1 << i, fillVal, 1);
			}
		}
		for (int i = 0; i < 31; i++) {
			printf("%d,%lld\n", 1 << i, times[i] / test_iters);
		}
		return 0;
	}

	auto start = chrono::high_resolution_clock::now();

	// This macro is used because the compiler throws a compile error when trying to parse 
	// the correct CUDA kernel syntax
	convEncode KERNEL_ARGS2((data_length + THREADS - 1) / THREADS, THREADS) (A, B);

	CUDACHECK(cudaDeviceSynchronize());

	auto end = chrono::high_resolution_clock::now() - start;

	long long elapsed_time = chrono::duration_cast<chrono::microseconds>(end).count();

	printf("Elapsed time %lld \u03bcs\n", elapsed_time);

	cudaFree(A);
	cudaFree(B);

	return 0;
}


