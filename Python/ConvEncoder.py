import numpy as np
import math
from functools import reduce
from time import time

# Implementation using numpy array operations
def encode_array(data, conv_filter):
    mod2 = np.vectorize(lambda x: x % 2)
    filt_w = conv_filter.shape[0]
    new_data = np.append(np.zeros(filt_w - 1), data)
    return mod2(np.matmul(
        new_data[np.stack([np.arange(i, i + filt_w) for i in range(data.shape[0])])],
        conv_filter).flatten().astype('uint8'))

def hamming_weight(n):
    s = 0
    while n:
        s += n & 1
        n >>= 1
    return s

def encode_state(data, conv_filters, filter_length):
    """
    Precomputes state transitions, then traverses data linearly.
    """
    states = []
    for i in range(1 << filter_length):
            states.append(int("".join(map(lambda x: str(hamming_weight(x & i) & 1), conv_filters)), 2))

    f = np.vectorize(lambda x: states[(data >> x) & ((1 << filter_length) - 1)])

    return reduce(lambda x,y: (x << 2) | y, map(int, f(np.arange(int(math.log2(data)), -1, -1))), 0)


# Filter polynomials, encoded as lists of integers
filters = [
    [0b111, 0b011]
]

# x = 0b1011010110110101101101011011010110110101
filt = [0b111,0b011]

# Really long binary string used for testing C program and comparing performance
testx = "010111010101001110000110111110110000101001100010101101111101111001111000111100100100101100" \
        "010011000111011100110011101001001111010111100100011100001010011111011000101101010100110111" \
        "010111010011100111011101011101011110110111011101010000011100001111111111100001001000101111" \
        "011100110011001010010010011100011100010010111001001000000100000001001000001010001101111011" \
        "111101010001101001011001011100010001001000110110010111000010111110011110110100100011101110" \
        "010100111100100101000111111000001100110101111011100010110000011001101010000111001101000100" \
        "001111010011111000011000111000010010101101100100110100011000010110000111000101001101100101" \
        "111011110011001001100001010101101111101110111010010001111001001010101010010100011000001000" \
        "110100100101101000001101100011010000001001001001011011111101111101000101110100001001000101" \
        "101000011011011101101001001001000001010101110110011110100111110010011110010101001100101011" \
        "011000101000110111111011110110101000001010111101001111101110001100100000001001111110000100" \
        "110101101010110011000011111101010110101101010010001001001111000111111111111111001110011101" \
        "000011101001001010110001111110011010100011110000000111001110000101010110101111011011101111" \
        "110010001110101110001101110000101000100011100001100001110110100000100111011000101110010111" \
        "100001001100111100011110110110011110101001100000011110010010110110111011111001000011011000" \
        "000111111001011111011100000001101011010000100111111010101110010111011100100011001000110000" \
        "100001111110110111001101011011111000011110011001011111010100100010001110001100011101010101" \
        "011000100111000111000011101001100111110001011100011011000011101101001011011100000100011000" \
        "001100110011111010110101000000010100110110101011001010010101010000101001100110011101011010" \
        "100000111110101111011100101010011011101100100010011101000011010000100011100100001110111000" \
        "001000110001111110100000011101101010100000001110000100010000101000000100100110001101100011" \
        "101010011111111011010001111001111011101000010101010101110010101010011001011100111100101011" \
        "100111110111111010111110001110000100001001110010101111001001111011110101101000011101010010" \
        "110000111000001110000100101100110001100111011100000101010101111110101010010111111001011011" \
        "110000110100000110100110110001010010110001111010010100111001101011011100111000101101110101" \
        "011000001100010111001111001001010110111000101011101011010001001101000110011000100100010101" \
        "100100010111001101110101001110001001110111101000011001000011011010100001000110011110100100" \
        "110110010111000001011000010101010010001011011010010011000111110011100010000001010100110101" \
        "110011111001000101000000110110111010001110100011111100100111011110110100011111000011110111" \
        "010000100001001110010010101010011001011100111110101011010100011100101111100101101101100000" \
        "111001101011111110001011111010011010111110011010000001011110010101000011010101111000101111" \
        "100110000011110110110111000001000011111010010100010000111100111010111001111111111101100000" \
        "100101000000100100010011011101011110001101010101001111011101111100001001001000000101110101" \
        "100011011001001000111101101010000110101100100010100000111001000111100010010011001101100111" \
        "000110111100000100110100111010000111010010000110011011101100010100101010010010110101010101" \
        "110011000100100000111001110001010000011010110100111000010110011100100110100111111110010100" \
        "011110011100010011110001100101000010110011011110001110000010101011000110101111111000011010" \
        "001000010101000110100010010100100001110111000001011100100010011110011001000010101111111011" \
        "110110110111011110111001110011101111101011010101011101001101000101110001110010100010111001" \
        "100010101010001111011001000011001100110001000011011101100010110001010100011000100111001000" \
        "001100110001011001101111100111101000100001110010010001110010011110111000011001100100010111" \
        "100000110110011000011100000000011011111011110111110101011000110100011011111001111011100000" \
        "010010110010000100101101100111010000010000100000011010100110100000000011000100110011010101" \
        "001000101010111010100001100110000100110101010010110100111010111111001010111100100101001111" \
        "101000011011100101100000000010100001101100010111000011101111110001001100001111010000100101" \
        "111111111101000011000111011001001011001011111100111110101100110100010001100010101011111001" \
        "111000111001101101011010101111010010011000011101111101110111101101111001100011110110000010" \
        "010011011111110100110101011010110100001110100010110110111010111110101100010011101110111000" \
        "100010011110000111100110011101010000001110000000000101011001100010011111100001011001000000" \
        "011100101100010010010100111001000101011111000100111001010011001101111101111010100101101010" \
        "001011010000110101011100101000010011000010011101111011000001111010111101100010101111000010" \
        "101110111111100101111010000001011110001001110000111110100000001111010011110000111001000101" \
        "011111010110000000001110110101110101010110001101001101011010001011011000100101001111101101" \
        "101111101001100100101111001010101110110101000100100010100010101000011010101111011101000101" \
        "111000011001010100100001000100110101101011001100101011011100010001010001100101001011001110" \
        "000110011111011111010010111011001010111101001110100001111010111011100000101001010000101000" \
        "010001010111111111101110111100010001111100000110000001001100010101001101110111100101001000" \
        "010001000001111101111111000010010101000011010111100110111011011000010001111110101000001110" \
        "101011010111011101001111001011010001110000110001111001101101101001010011100110000111011000" \
        "110001110110001101001011011011010100001111001101000110101110110011100101011001010010101000" \
        "110111001111000100010110100001101100101110110100011000011100011110110010101110100011010111" \
        "000010001110011010101000101100011101110000000111100111110011111001000000100011001101101101" \
        "111101011000101010001000000110101111000000101101011001100110110010111100101110000001101000" \
        "000111100011111010010100011011010000111011011010101010000110100011001000001000110010000001" \
        "000101001011011000000101100000011001001110110111100001011010110000010001000100010111101101" \
        "001110010001100101001011000010101011001001011110000111101001011001000101101001111000000101" \
        "100001101010001101110110011011110011110001111011110111111101011011101110100110100011010110" \
        "111010011110010110001101101111011011101001110001001101001100010101101100111010111111000101" \
        "111010100101010110101110001110100011100110011011111011000000001100011111010001100111011101" \
        "110100000101110011101110001111010110000111101000101111011001110110011010001110000111011101" \
        "001101001010101010111000111011101100100101001100001010101110001111001101111111111001000110" \
        "010100111010101011101110011100111100011100010111100010111110111111010111000001101000100000" \
        "010001000100010100100110010111101000000101101001111100000011111101010110001111111100000011" \
        "100111100101110101001011011011010001001011100010011101010110010001111000101110110111010100" \
        "000001101000000011000011101011101100001101100111101010111111000010011101011110111011011111" \
        "111111100011001100011110100110010111011011000110000000011011011010111111010101100110011010" \
        "000100101110000100111011101000000110001011000011011000010100001011111110100110100110010111" \
        "110111011011101010100111110000111110010111000100101001110001111101001111001001100000100010" \
        "001010101110111011010000111101010110001001001111010101111000110111110111101110101110101100" \
        "001110001001001001110001000010101010010101001111010111111100001001101010010001011011111010" \
        "111010001011011001001000001101101000110000101011001000111010100011100101101010100100110101" \
        "110111111000111011111101001001100000010000011010111010010010010101111110110111010111111100" \
        "010100000010110101000111101001101000100000100000011001101001010011011100000110010110111001" \
        "011010110000011111001010011110000101011100001001010011001110100100111101001001111110010010" \
        "000010011011100110010000111111010101110111001010010101111000111111100000000000111100011001" \
        "110111001111101110000100101001111001001011111000101000000000100110000000110100110110001011" \
        "011100001100000001110011000000010000110011101111011010011001110100000001001011011000011010" \
        "011010101000111010001001111101100000100000111010110001110100111000111011001100010101101001" \
        "101000110111101110100010100111101101110101110110001010101000010101111000000101001001100100" \
        "111001110000001101111110011000110011100111110000101000110101010011010101010111001101100011" \
        "001011011101100011110110110110101100111101011110010011011000010101000001100001100011001010" \
        "001101010110010101001011011100110100101011110001101111101010101011111110000111001100111010" \
        "111101010111101110000000010001111000100101110100100111011111110000100110101110001011111010" \
        "001110100110110111100100110000110000010000010010000011000101100011010001110000011011000100" \
        "101101000110111101110011010000011101100011101011101001110100101010100001010001101001011110" \
        "001000000111101010010010000101111000100100001100100100001000111100101111010000000100010101" \
        "011111001001001001110010110110101000100010001000010011111110011101001100111011100010000000" \
        "010100101001111110101111110011111000101111010011011011000011101001101110100011000111010101" \
        "100111110011010111001011110110111111101101000110111100001100101100000111111010110001010001" \
        "010011101110001101111010010000010011011010100101011001100011011001001000001001100111101011" \
        "101111001001111100110111010111001001101011000000000111011001011101000011110001101000101001" \
        "101101011010110100110000110010101100000000001101001000110100110101101111100010111010101010" \
        "111000011010001111010101101101001101010100000100100000011100000000111011011100111101101100" \
        "100111100001010000011000011010011110001111101000000110101110010100111010011000010100111101" \
        "000100101110100001010001011101100100001110111001100110011011111001010011100110100011111111" \
        "111101011001010001001011100100111001110001110101101000100011001011110110010000001010010010" \
        "000011000101101110011000001101001111100000001111011111010010101000111101111010100001001001" \
        "001000011010011101101010111101010100001101110010000010000110111001111111111100010110111101" \
        "000111010111101001011111111010110001101101100000100100011010001101110111101100111010000110" \
        "011000010101011000011001100110011111010010111101011001010111100000110111111110100110010000" \
        "111001111010011110100101111101010011000101000001001000011101100011100110110111110011001000" \
        "1110111001"

# Benchmark performance
start = time()
for i in range(1000):
    encode_state(int(testx, 2), filt, 3)

print(time() - start)
