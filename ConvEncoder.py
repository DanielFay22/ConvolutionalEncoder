import numpy as np
import math
from functools import reduce
from collections import defaultdict
from time import time


def new_dict(order, def_type):
    """
    Create a default dictionary of the specified dimension and type.
    """
    if order == 1:
        return defaultdict(def_type)
    elif order > 1:
        return defaultdict(lambda: new_dict(order - 1, def_type))

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
    Precomputes state transitions, then traverses data.
    """
    states = []
    for i in range(1 << filter_length):
            states.append(int("".join(map(lambda x: str(hamming_weight(x & i) & 1), conv_filters)), 2))
                     # 1: int("".join(map(lambda x: str(hamming_weight(x & ((i << 1) + 1)) & 1), conv_filters)), 2)}

    f = np.vectorize(lambda x: states[(data >> x) & ((1 << filter_length) - 1)])

    return reduce(lambda x,y: (x << 2) | y, map(int, f(np.arange(int(math.log2(data)), -1, -1))), 0)


def normalize(array):
    return np.divide(array, array.sum(-1))#.reshape((-1, 1))

def decode_conv(data):
    initial_prob = np.ones((2,2)) / 4
    trans_mat = np.ones((2, 2, 2)) / 2
    E = np.ones((2,4)) / 4
    states = np.arange(0,2)

    # print(E)
    # print(trans_mat)
    # print(initial_prob)

    bigrams = [(s1, s2) for s1 in states for s2 in states]

    data_tokens = np.array([((data >> i) & 0b11) for i in range(int(math.log2(data)) - 1, -1, -2)])

    v = np.zeros((1, 2, 2))
    bp = np.zeros((1,2,2))

    for s1, s2 in bigrams:
        v[0, s1, s2] = initial_prob[s1, s2] * E[s1, data_tokens[0]] * E[s2, data_tokens[1]]

    for i in range(1, data_tokens.shape[0]):
        v = np.concatenate((v, np.array([[[E[s2, data_tokens[i]] * max(
            [v[i-1, s3, s2] * trans_mat[s3, s2, s1] for s3 in states]) for s1 in states]
                                         for s2 in states]])))
        bp = np.concatenate((bp, np.array([[[max([(v[i - 1, s3, s2] * trans_mat[s3, s2, s1], s3) for s3 in states])[1]
                                           for s1 in states] for s2 in states]])))

    z = np.zeros(data_tokens.shape, dtype='int')

    z[-2], z[-1] = max([(v[-1, s1, s2], (s1, s2)) for s1, s2 in bigrams])[1]
    for i in range(data_tokens.shape[0] - 3, -1, -1):
        z[i] = bp[i + 2, z[i+1], z[i + 2]]

    return z




filters = [
    np.array([[1,1],[1,0],[1,1]]),
    np.array([[1,1],[1,0],[1,1],[1,1]]),
    np.array([[1,1],[0,1],[1,0],[1,0],[1,1]])
]

x = 0b1011010110110101101101011011010110110101
filt = [0b111,0b011]

print(bin(encode_state(x, filt, 3)))
print(encode_state(x, filt, 3))
# print(decode_conv(encode_state(x, filt, 3)))
# print(bin(x))

# print(bin(encode_state(0b1011,[0b111,0b011],3)))
# np.array(list(map(int, bin(x)[2:])))

# print(bin(encode_state(0b1011, [0b111,0b011], 3)))
# print(make_trans([0b111,0b011], 3))

# x_arr = np.random.randint(0,2,100000)
# x = int("".join(map(str, list(x_arr))), 2)
#
# print(x)
#
# start = time()
#
# int("".join(map(str, list(encode_array(x_arr,np.array([[1,0],[1,1],[1,1]]))))), 2)
# print(time() - start)
#
# start = time()
# encode_state(x, [0b111,0b011], 3)
# print(time() - start)