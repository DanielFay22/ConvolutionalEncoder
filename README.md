# ConvolutionalEncoder

* The python file contains a few implementations, as well as a variety of tests of those methods.
* The C implementation does not use CUDA. However, it is designed to mimic the execution of a CUDA kernel, using `idx` and `start` to determine position in the array. In the CUDA implementation, these variables would be directly derived from the thread index and block index, respectively.
