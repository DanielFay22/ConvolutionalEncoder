#include <iostream>
#include <time.h>
#include <chrono>
#include <cstring>

using namespace std;

#define NUM_BITS (8 * sizeof(unsigned int))

unsigned int data_length = (1 << 23);
const unsigned int fillVal = 0x0F0F0F0F;

//const unsigned int THREADS = 256;
const unsigned int FILTER_LENGTH = 3;
const unsigned int NUM_FILTERS = 2;
const unsigned int FILTERS = (0b111 << FILTER_LENGTH) | 0b011;

char nums[2] = { '0', '1' };

#define dataArrayLength ((data_length + NUM_BITS - 1) / NUM_BITS)
#define resultArrayLength ((data_length * NUM_FILTERS + NUM_BITS - 1) / NUM_BITS)

void printBitString(const unsigned int *data, int length) {
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < NUM_BITS; j++) {
            cout << nums[(data[i] >> (NUM_BITS - 1 - j)) & 1];
        }
    }
    cout << endl;
}

// This function is specifically structured to mimic the execution of a CUDA kernel,
// which is why the entire operation is contained in two for loops.
// The variables "start" and "idx" will be derived from the block index and thread index, respectively.
void convEncode(const unsigned int *data, unsigned int *output) {
    for (unsigned int start = 0; start < (data_length + NUM_BITS - 1) / (NUM_BITS * 2); start++)   {
        for (unsigned int idx = 0; idx < 2 * NUM_BITS; idx++)  {

            // Get the relevant bits from the data, shifted to the FILTER_LENGTH LSBs
            unsigned int val = (((idx / NUM_BITS) | start
                        ? data[idx / NUM_BITS + start - 1] << (1 + (idx % NUM_BITS)) : 0)
                                | (data[idx / NUM_BITS + start] >> (NUM_BITS - 1 - (idx % NUM_BITS))))
                                        & ((1 << FILTER_LENGTH) - 1 );

            #pragma unroll
            for (unsigned int i = 0; i < NUM_FILTERS; i++) {

                // Apply the relevant filter to the data
                unsigned int n = val & ((FILTERS >> ((NUM_FILTERS - i - 1) * FILTER_LENGTH))
                        & ((1 << FILTER_LENGTH) - 1 ));

                // Calculate the parity of the result. This works by first computing
                // the parity of each pair of bits, then the parity of every 4 bits,
                // and finally using a multiply to combine all the results at the 28th bit.
                n ^= n >> 1;                            // Parity of pairs of bits
                n ^= n >> 2;                            // Parity of every 4 bits
                n = (n & 0x11111111U) * 0x11111111U;    // Multiply to combine all the 4 bit parities
                n = (n >> 28) & 1;                      // Extract the final result

                // Put the result bit in the correct place in the array
                output[NUM_FILTERS * start + (NUM_FILTERS * idx) / NUM_BITS] |=
                        (n << ((NUM_BITS * NUM_FILTERS - 1 - (NUM_FILTERS * (idx % NUM_BITS) + i))));
            }
        }
    }
}


long long runTest(unsigned int length)  {
    data_length = length;

    unsigned int *input;
    unsigned int *output;

    input = (unsigned int*)(malloc(dataArrayLength * sizeof(unsigned int)));
    output = (unsigned int*)(malloc(resultArrayLength * sizeof(unsigned int)));

    for (unsigned int i = 0; i < dataArrayLength; i++)   { input[i] = fillVal; }
    for (unsigned int i = 0; i < resultArrayLength; i++) { output[i] = 0; }

    auto start = chrono::high_resolution_clock::now();

    convEncode(input, output);

    auto ends = chrono::high_resolution_clock::now();

    long long cpu_time_used = chrono::duration_cast<chrono::microseconds>(ends - start).count();

    return cpu_time_used;
}


int main() {
    unsigned int *input;
    unsigned int *output;

    input = (unsigned int*)(malloc(dataArrayLength * sizeof(unsigned int)));
    output = (unsigned int*)(malloc(resultArrayLength * sizeof(unsigned int)));

    memset(input, fillVal, sizeof(unsigned int) * dataArrayLength);
    memset(output, 0, sizeof(unsigned int) * resultArrayLength);

//    for (unsigned int i = 0; i < dataArrayLength; i++)   { input[i] = fillVal; }
//    for (unsigned int i = 0; i < resultArrayLength; i++) { output[i] = 0; }

    long long times[32];
    for (unsigned int i = 0; i < 32; i ++)   {
        times[i] = runTest((unsigned int)(1 << i));
        printf("%d,%lld\n", 1 << i, times[i]);
    }



//    chrono::time_point<chrono::high_resolution_clock> start = chrono::high_resolution_clock::now();
//
//    convEncode(input, output);
//
//    chrono::time_point<chrono::high_resolution_clock> ends = chrono::high_resolution_clock::now();
//
//    long long cpu_time_used = chrono::duration_cast<chrono::microseconds>(ends - start).count(); //((double)(end - start)) / (CLOCKS_PER_SEC / 1e6);
//
////    printf("%d\n", CLOCKS_PER_SEC);
////    printf("Execution required %lld us\n", cpu_time_used);
//
//    printf("%d,%lld", data_length, cpu_time_used);

//    printf("%x\n%x\n", output[0], output[1]);
//    printBitString(output, resultArrayLength);
}
