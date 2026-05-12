# PCA: EXP-1  SUM ARRAY GPU
<h3>ENTER YOUR NAME: G.HINDHU</h3>
<h3>ENTER YOUR REGISTER NO: 212223230079</h3>
<h3>EX. NO : 1</h3>
<h3>DATE: 12.05.26</h3>
<h1> <align=center> SUM ARRAY ON HOST AND DEVICE </h3>
PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## AIM:

To perform vector addition on host and device.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## PROGRAM:
TYPE YOUR CODE HERE
```
!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter
```
### THREAD=512
```
%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                    \
{                                                                      \
    const cudaError_t error = call;                                    \
    if (error != cudaSuccess)                                          \
    {                                                                  \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);         \
        fprintf(stderr, "code: %d, reason: %s\n",                      \
                error, cudaGetErrorString(error));                     \
        exit(1);                                                       \
    }                                                                  \
}

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);

    return ((double)tp.tv_sec +
           (double)tp.tv_usec * 1.e-6);
}

#endif

// check GPU result with CPU result
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = false;

            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at index %d\n",
                   hostRef[i], gpuRef[i], i);

            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
}

// initialize array
void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// CPU vector addition
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// GPU kernel
__global__ void sumArraysOnGPU(float *A, float *B,
                               float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set device
    int dev = 0;

    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    printf("Using Device %d: %s\n",
           dev, deviceProp.name);

    CHECK(cudaSetDevice(dev));

    // vector size
    int nElem = 1 << 24;

    printf("Vector size %d\n", nElem);

    size_t nBytes = nElem * sizeof(float);

    // allocate host memory
    float *h_A, *h_B, *hostRef, *gpuRef;

    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    double iStart, iElaps;

    // initialize host data
    iStart = seconds();

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    iElaps = seconds() - iStart;

    printf("initialData Time elapsed %f sec\n",
           iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // CPU addition
    iStart = seconds();

    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    iElaps = seconds() - iStart;

    printf("sumArraysOnHost Time elapsed %f sec\n",
           iElaps);

    // allocate device memory
    float *d_A, *d_B, *d_C;

    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_B, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    // copy data to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes,
                     cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(d_B, h_B, nBytes,
                     cudaMemcpyHostToDevice));

    // kernel configuration
    int iLen = 512;

    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    // launch kernel
    iStart = seconds();

    sumArraysOnGPU<<<grid, block>>>(d_A, d_B,
                                    d_C, nElem);

    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;

    printf("sumArraysOnGPU <<<%d, %d>>> Time elapsed %f sec\n",
           grid.x, block.x, iElaps);

    // check kernel errors
    CHECK(cudaGetLastError());

    // copy result back
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes,
                     cudaMemcpyDeviceToHost));

    // verify result
    checkResult(hostRef, gpuRef, nElem);

    // free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
```
## THREAD=1023
```
%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                    \
{                                                                      \
    const cudaError_t error = call;                                    \
    if (error != cudaSuccess)                                          \
    {                                                                  \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);         \
        fprintf(stderr, "code: %d, reason: %s\n",                      \
                error, cudaGetErrorString(error));                     \
        exit(1);                                                       \
    }                                                                  \
}

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);

    return ((double)tp.tv_sec +
           (double)tp.tv_usec * 1.e-6);
}

#endif

// check GPU result with CPU result
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = false;

            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at index %d\n",
                   hostRef[i], gpuRef[i], i);

            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
}

// initialize array
void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// CPU vector addition
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// GPU kernel
__global__ void sumArraysOnGPU(float *A, float *B,
                               float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set device
    int dev = 0;

    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    printf("Using Device %d: %s\n",
           dev, deviceProp.name);

    CHECK(cudaSetDevice(dev));

    // vector size
    int nElem = 1 << 24;

    printf("Vector size %d\n", nElem);

    size_t nBytes = nElem * sizeof(float);

    // allocate host memory
    float *h_A, *h_B, *hostRef, *gpuRef;

    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    double iStart, iElaps;

    // initialize host data
    iStart = seconds();

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    iElaps = seconds() - iStart;

    printf("initialData Time elapsed %f sec\n",
           iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // CPU addition
    iStart = seconds();

    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    iElaps = seconds() - iStart;

    printf("sumArraysOnHost Time elapsed %f sec\n",
           iElaps);

    // allocate device memory
    float *d_A, *d_B, *d_C;

    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_B, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    // copy data to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes,
                     cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(d_B, h_B, nBytes,
                     cudaMemcpyHostToDevice));

    // kernel configuration
    int iLen = 1023;

    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    // launch kernel
    iStart = seconds();

    sumArraysOnGPU<<<grid, block>>>(d_A, d_B,
                                    d_C, nElem);

    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;

    printf("sumArraysOnGPU <<<%d, %d>>> Time elapsed %f sec\n",
           grid.x, block.x, iElaps);

    // check kernel errors
    CHECK(cudaGetLastError());

    // copy result back
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes,
                     cudaMemcpyDeviceToHost));

    // verify result
    checkResult(hostRef, gpuRef, nElem);

    // free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
```
## THREAD= 1024
```
%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                    \
{                                                                      \
    const cudaError_t error = call;                                    \
    if (error != cudaSuccess)                                          \
    {                                                                  \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);         \
        fprintf(stderr, "code: %d, reason: %s\n",                      \
                error, cudaGetErrorString(error));                     \
        exit(1);                                                       \
    }                                                                  \
}

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);

    return ((double)tp.tv_sec +
           (double)tp.tv_usec * 1.e-6);
}

#endif

// check GPU result with CPU result
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = false;

            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at index %d\n",
                   hostRef[i], gpuRef[i], i);

            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
}

// initialize array
void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// CPU vector addition
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// GPU kernel
__global__ void sumArraysOnGPU(float *A, float *B,
                               float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set device
    int dev = 0;

    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    printf("Using Device %d: %s\n",
           dev, deviceProp.name);

    CHECK(cudaSetDevice(dev));

    // vector size
    int nElem = 1 << 24;

    printf("Vector size %d\n", nElem);

    size_t nBytes = nElem * sizeof(float);

    // allocate host memory
    float *h_A, *h_B, *hostRef, *gpuRef;

    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    double iStart, iElaps;

    // initialize host data
    iStart = seconds();

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    iElaps = seconds() - iStart;

    printf("initialData Time elapsed %f sec\n",
           iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // CPU addition
    iStart = seconds();

    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    iElaps = seconds() - iStart;

    printf("sumArraysOnHost Time elapsed %f sec\n",
           iElaps);

    // allocate device memory
    float *d_A, *d_B, *d_C;

    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_B, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    // copy data to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes,
                     cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(d_B, h_B, nBytes,
                     cudaMemcpyHostToDevice));

    // kernel configuration
    int iLen = 1024;

    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    // launch kernel
    iStart = seconds();

    sumArraysOnGPU<<<grid, block>>>(d_A, d_B,
                                    d_C, nElem);

    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;

    printf("sumArraysOnGPU <<<%d, %d>>> Time elapsed %f sec\n",
           grid.x, block.x, iElaps);

    // check kernel errors
    CHECK(cudaGetLastError());

    // copy result back
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes,
                     cudaMemcpyDeviceToHost));

    // verify result
    checkResult(hostRef, gpuRef, nElem);

    // free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
```
## THREAD=256
```
%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                    \
{                                                                      \
    const cudaError_t error = call;                                    \
    if (error != cudaSuccess)                                          \
    {                                                                  \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);         \
        fprintf(stderr, "code: %d, reason: %s\n",                      \
                error, cudaGetErrorString(error));                     \
        exit(1);                                                       \
    }                                                                  \
}

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);

    return ((double)tp.tv_sec +
           (double)tp.tv_usec * 1.e-6);
}

#endif

// check GPU result with CPU result
void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = true;

    for (int i = 0; i < N; i++)
    {
        if (fabs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = false;

            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at index %d\n",
                   hostRef[i], gpuRef[i], i);

            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
}

// initialize array
void initialData(float *ip, int size)
{
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// CPU vector addition
void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// GPU kernel
__global__ void sumArraysOnGPU(float *A, float *B,
                               float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set device
    int dev = 0;

    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    printf("Using Device %d: %s\n",
           dev, deviceProp.name);

    CHECK(cudaSetDevice(dev));

    // vector size
    int nElem = 1 << 24;

    printf("Vector size %d\n", nElem);

    size_t nBytes = nElem * sizeof(float);

    // allocate host memory
    float *h_A, *h_B, *hostRef, *gpuRef;

    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    double iStart, iElaps;

    // initialize host data
    iStart = seconds();

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    iElaps = seconds() - iStart;

    printf("initialData Time elapsed %f sec\n",
           iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // CPU addition
    iStart = seconds();

    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    iElaps = seconds() - iStart;

    printf("sumArraysOnHost Time elapsed %f sec\n",
           iElaps);

    // allocate device memory
    float *d_A, *d_B, *d_C;

    CHECK(cudaMalloc((float **)&d_A, nBytes));
    CHECK(cudaMalloc((float **)&d_B, nBytes));
    CHECK(cudaMalloc((float **)&d_C, nBytes));

    // copy data to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes,
                     cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(d_B, h_B, nBytes,
                     cudaMemcpyHostToDevice));

    // kernel configuration
    int iLen = 256;

    dim3 block(iLen);
    dim3 grid((nElem + block.x - 1) / block.x);

    // launch kernel
    iStart = seconds();

    sumArraysOnGPU<<<grid, block>>>(d_A, d_B,
                                    d_C, nElem);

    CHECK(cudaDeviceSynchronize());

    iElaps = seconds() - iStart;

    printf("sumArraysOnGPU <<<%d, %d>>> Time elapsed %f sec\n",
           grid.x, block.x, iElaps);

    // check kernel errors
    CHECK(cudaGetLastError());

    // copy result back
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes,
                     cudaMemcpyDeviceToHost));

    // verify result
    checkResult(hostRef, gpuRef, nElem);

    // free device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
```

## OUTPUT:
<img width="1752" height="239" alt="image" src="https://github.com/user-attachments/assets/4f1c6425-1247-4564-abba-b0a248411d17" />

## THREAD=512
<img width="1558" height="260" alt="image" src="https://github.com/user-attachments/assets/c29f35f1-7cba-4684-8e92-015b079c2115" />

## THREAD=1023
<img width="1797" height="293" alt="image" src="https://github.com/user-attachments/assets/b79dec33-32fe-4c02-b232-766732c41458" />

## THREAD=1024
<img width="1612" height="237" alt="image" src="https://github.com/user-attachments/assets/0ab76512-69e4-4be5-8931-d1a616197350" />

## THREAD=256
<img width="1768" height="246" alt="image" src="https://github.com/user-attachments/assets/c704ba47-6fa6-42b9-bf4c-68494bb2f0ad" />

## RESULT:


Among all thread configurations, THREAD = 1024 gives the best efficiency because it has the minimum execution time (≈ 0.000911 sec). Therefore, the vector addition operation executes faster with 1024 threads compared to 256, 512, and 1023 threads.Thus, Implementation of sum arrays on host and device is done in nvcc cuda using random number.
