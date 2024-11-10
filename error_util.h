#include <iostream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

#define FatalError(s) do {                                             \
    std::cout << std::flush << "ERROR: " << s << " in " <<             \
              __FILE__ << ':' << __LINE__ << "\nAborting...\n";        \
    cudaDeviceReset();                                                 \
    exit(-1);                                                          \
} while (0)

// check CUDA error
#define checkCUDA(status) do {                                              \
    std::stringstream _err;                                                 \
    if(status != cudaSuccess){                                                        \
        _err << "Cuda Failure (" << cudaGetErrorString(status) << ')';     \
        FatalError(_err.str());                                             \
    }                                                                       \
} while(0) 

// check CUDNN error
#define checkCUDNN(status) do {                                             \
    std::stringstream _err;                                                 \
    if (status != CUDNN_STATUS_SUCCESS) {                                   \
        _err << "CUDNN FAILURE (" << cudnnGetErrorString(status) << ')';    \
        FatalError(_err.str());                                             \
    }                                                                       \
} while(0)

// check CUBLAS error
#define checkCUBLAS(status) do {                                                \
    std::stringstream _err;                                                     \
    if (status != CUBLAS_STATUS_SUCCESS) {                                      \
        _err << "CUBLAS FAILURE (code =" << status << ')';      \
        FatalError(_err.str());                                                  \
    }                                                                           \
} while(0)
