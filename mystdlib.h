/*
 * =====================================================================================
 *
 *       Filename:  mystdlib.h
 *
 *    Description:  Contains my own lib
 *
 *        Version:  1.0
 *        Created:  12/26/2013 08:29:15 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Hoang-Ngan Nguyen (), zhoangngan-gmail
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef  MYSTDLIB_H_
#define  MYSTDLIB_H_
#include <cuda.h>
#include <math_functions.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <assert.h>
#include <eigen3/Eigen/Dense>

// assert() is only supported on devices of compute capability 2.0 or higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef assert
#define assert(arg)
#endif

#define PI   3.14159265358979324
#define SPI  1.77245385090551603
#define eps  2.22044604925031e-16

// This function is copied from book.h: CUDA by example.
  static void 
HandleError( 
    cudaError_t err,
    const char *file,
    int line 
  ) 
{
  if( err != cudaSuccess ) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
            file, line );
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void
kernelConfig (uint & dimGrid, uint & dimBlock, const size_t & size);

enum deviceType { HOST, GPU };

template <deviceType device> class Matrix;

typedef Matrix <HOST> MatrixOnHost;
typedef Matrix <GPU>  MatrixOnDevice;

// === dot: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  dot
//  Description:  Return the dot product of two vectors.
// =============================================================================
__device__ __host__ double
dot ( /* argument list: {{{*/
    const double* const x, const double* const y, const int& N = 3 
    ) /* ------------- end of argument list -----------------------------------}}}*/ 
{ /* dot implementation: {{{*/
  double r = 0;
  for (int i = 0; i < N; i++) 
    r += x[i]*y[i];
  return r;
} /*}}}*/
/* ---------------  end of DEVICE function dot  -------------- }}}*/

__device__ __host__ double 
experf (double x, double z, const double &c = 2) {/*{{{*/
  double a = 0;
  if ((x*z < 0) || (x*x + z*z < 700))
    a = exp(c*x*z) * erfc(z + x);
  return a;
}/*}}}*/

template <deviceType device>
void zMalloc (void** ptr, size_t size);

template <deviceType device>
void zFree (void* ptr);

template <deviceType device>
void zMemcpy (void* to, void* from, size_t size);

template <deviceType toDevice, deviceType fromDevice>
void zMemcpy (void* to, void* from, size_t size); 

__global__ void
kernelMemcpy (void* to, void* from, size_t size);

template <typename T> __global__ void
kernelSum (T* retval, T* data, size_t size);

template <typename T> __global__ void
kernelDot (T* dot, T* a, T* b, size_t size);

template <deviceType device>
void zMemset ( double* ptr, double value, size_t size);

template <typename T> 
__device__ __host__ void
cross (T* result, T* a, T* b) {
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];
}

template < typename T > T
exactSum ( T* data, size_t size);

template < typename T > 
__device__ __host__ void
swap (T & x, T & y) {
  T temp = x;
  x = y;
  y = temp;
}

uint 
chooseSize (size_t size, int bit = 10) {/*{{{*/
  if (size >= (1 << bit)) return (1 << bit);
  if (size == 0 ) return 0;
  uint end = bit, start = 0, pos;
  while (start < end) {
    pos = (end + start) / 2;
    if (size >= (1 << pos)) start = pos + 1;
    if (size <= (1 << pos)) end = pos;
  }
  if (size < (1 << pos)) pos--;
  return (1 << pos);
}/*}}}*/

// === kernelMemcpy: KERNEL FUNCTION  =========================================={{{ 
//         Name:  kernelMemcpy
//  Description:  Copy data between two device memory locations.
// =============================================================================
__global__ void
kernelMemcpy ( /* argument list: {{{*/
    void* gto, void* gfrom, size_t size
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* kernelMemcpy implementation: {{{*/
  size_t dsize = size / sizeof(size_t);
  size_t osize = size % sizeof(size_t);
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;   
  size_t gridSize = blockDim.x * gridDim.x;
  size_t* to = (size_t*)gto;
  size_t* from = (size_t*)gfrom;
  char* cto  = (char*)(&to[dsize]);
  char* cfrom  = (char*)(&from[dsize]);
  
  while (tid < dsize) {
    to[tid] = from[tid];
    tid += gridSize;
  }

  if (osize == 0) return;

  // remaining data of size < size_t
  tid = threadIdx.x + blockIdx.x * blockDim.x;   
  while (tid < osize) {
    cto[tid] = cfrom[tid];
    tid += gridSize;
  }

} /*}}}*/
/* ---------------  end of function kernelMemcpy  -------------------- }}}*/

// === kernelDot: TEMPLATE FUNCTION  =========================================={{{ 
//         Name:  kernelDot
//  Description:  Compute a PARTIAL dot product between two vectors.
// =============================================================================
template <typename T> __global__ void
kernelDot ( /* argument list: {{{*/
    T* d, T* x, T* y, size_t size 
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* kernelDot implementation: {{{*/
  __shared__ T a[512];
  uint lid = threadIdx.x;
  uint blockSize = blockDim.x;
  uint i = blockIdx.x + blockIdx.y * gridDim.x;
  i = lid + i * blockSize;
  uint gridSize = blockSize * gridDim.x * gridDim.y;

  a[lid] = 0;
  
  while (i < size) {
    a[lid] += x[i] * y[i];
    i += gridSize;
  }
  __syncthreads();

  if (blockSize >= 512) {
    if (lid < 256)
      a[lid] += a[lid + 256];
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (lid < 128)
      a[lid] += a[lid + 128];
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (lid < 64)
      a[lid] += a[lid + 64];
    __syncthreads();
  }
  if (lid < 32) {
    if (blockSize >= 64) a[lid] += a[lid + 32];
    if (blockSize >= 32) a[lid] += a[lid + 16];
    if (blockSize >= 16) a[lid] += a[lid + 8];
    if (blockSize >= 8)
      a[lid] += a[lid + 4];
    if (blockSize >= 4)
      a[lid] += a[lid + 2];
    if (blockSize >= 2)
      a[lid] += a[lid + 1];
  }

  if (lid == 0) {
    d[blockIdx.x + blockIdx.y * gridDim.x] = a[0];
//    printf("gridSize = %u, blockSize = %u, a[0] = %+20.16e\n", \
//        gridSize, blockSize, a[0]);
  }
} /*}}}*/
/* ---------------  end of function kernelDot  -------------------- }}}*/

// === kernelSum: TEMPLATE FUNCTION  =========================================={{{ 
//         Name:  kernelSum
//  Description:  Using sum reduction to give the sum of all elements of a
//  matrix.
// =============================================================================
template <typename T> __global__ void
kernelSum ( /* argument list: {{{*/
    T* retval, T* data, size_t size
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /*  implementation: {{{*/
  __shared__ T a[512];
  uint lid = threadIdx.x;
  uint blockSize = blockDim.x;
  uint i = blockIdx.x + blockIdx.y * gridDim.x;
  i = lid + i * 2 * blockSize;
  uint gridSize = 2 * blockSize * gridDim.x * gridDim.y;
  assert( size >= gridSize );

  a[lid] = 0;
  
  while (i < size) {
    if (i + blockSize < size) {
      a[lid] += data[i] + data[i + blockSize];
    }
    else {
      a[lid] += data[i];
    }
    i += gridSize;
  }
  __syncthreads();

  if (blockSize >= 512) {
    if (lid < 256)
      a[lid] += a[lid + 256];
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (lid < 128)
      a[lid] += a[lid + 128];
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (lid < 64)
      a[lid] += a[lid + 64];
    __syncthreads();
  }
  if (lid < 32) {
    if (blockSize >= 64) a[lid] += a[lid + 32];
    if (blockSize >= 32) a[lid] += a[lid + 16];
    if (blockSize >= 16) a[lid] += a[lid + 8];
    if (blockSize >= 8)
      a[lid] += a[lid + 4];
    if (blockSize >= 4)
      a[lid] += a[lid + 2];
    if (blockSize >= 2)
      a[lid] += a[lid + 1];
  }

  if (lid == 0) {
    retval[blockIdx.x + blockIdx.y * gridDim.x] = a[0];
//    printf("gridSize = %u, blockSize = %u, a[0] = %+20.16e\n", \
//        gridSize, blockSize, a[0]);
  }
} /*}}}*/
/* ---------------  end of function kernelSum  -------------------- }}}*/

// === kernelMax: TEMPLATE FUNCTION  =========================================={{{ 
//         Name:  kernelMax
//  Description:  Return the max of the absolute value of each element of an
//  array. The size of the array must be larger than or equal to
//  gridSize = 2 * blockSize * gridDim.x * gridDim.y;
// =============================================================================
template <typename T>
__global__ void
kernelMax ( /* argument list: {{{*/
    T* retval, T* data, size_t size
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /*  implementation: {{{*/
  __shared__ T a[512];
  uint lid = threadIdx.x;
  uint blockSize = blockDim.x;
  uint i = blockIdx.x + blockIdx.y * gridDim.x;
  i = lid + i * 2 * blockSize;
  uint gridSize = 2 * blockSize * gridDim.x * gridDim.y;
  assert( size >= gridSize );

  T temp;

  if (i + blockSize < size)
    a[lid] = (data[i] >= data[i + blockSize]) ? data[i] : data[i + blockSize];
  i += gridSize;
//  else
//    if (i < size) 
//      a[lid] = data[i];
//    else
//      a[lid] = data[size-1];

  while (i < size) {
    if (i + blockSize < size) {
      temp = (data[i] >= data[i + blockSize]) ? data[i] : data[i + blockSize];
      if (a[lid] < temp) a[lid] = temp;
    }
    else {
      if (a[lid] < data[i]) a[lid] = data[i];
    }
    i += gridSize;
  }
  __syncthreads();

  if (blockSize >= 512) {
    if (lid < 256)
      if (a[lid] < a[lid + 256]) a[lid] = a[lid + 256];
    __syncthreads();
  }
  if (blockSize >= 256) {
    if (lid < 128)
      if (a[lid] < a[lid + 128]) a[lid] = a[lid + 128];
    __syncthreads();
  }
  if (blockSize >= 128) {
    if (lid < 64)
      if (a[lid] < a[lid + 64]) a[lid] = a[lid + 64];
    __syncthreads();
  }
  if (lid < 32) {
    if (blockSize >= 64)
      if (a[lid] < a[lid + 32]) a[lid] = a[lid + 32];
    if (blockSize >= 32)
      if (a[lid] < a[lid + 16]) a[lid] = a[lid + 16];
    if (blockSize >= 16)
      if (a[lid] < a[lid + 8]) a[lid] = a[lid + 8];
    if (blockSize >= 8)
      if (a[lid] < a[lid + 4]) a[lid] = a[lid + 4];
    if (blockSize >= 4)
      if (a[lid] < a[lid + 2]) a[lid] = a[lid + 2];
    if (blockSize >= 2)
      if (a[lid] < a[lid + 1]) a[lid] = a[lid + 1];
  }

  if (lid == 0) {
    retval[blockIdx.x + blockIdx.y * gridDim.x] = a[0];
//    printf("gridSize = %u, blockSize = %u, a[0] = %+20.16e\n", \
//        gridSize, blockSize, a[0]);
  }
} /*}}}*/
/* ---------------  end of function kernelMax  -------------------- }}}*/

// === kernelSubtract: TEMPLATE FUNCTION  =========================================={{{ 
//         Name:  kernelSubtract
//  Description:  The difference of two vectors.
// =============================================================================
template <typename T>
__global__ void
kernelSubtract ( /* argument list: {{{*/
    T* result, T* lhs, T* rhs, size_t size 
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /*  implementation: {{{*/
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  size_t gridSize = blockDim.x * gridDim.x;
  
  while (tid < size) {
    result[tid] = lhs[tid] - rhs[tid];
    tid += gridSize;
  }
} /*}}}*/
/* ---------------  end of function kernelSubtract  -------------------- }}}*/

// === kernelMulScalar: TEMPLATE FUNCTION  =========================================={{{ 
//         Name:  kernelMulScalar
//  Description:  Multiply an array with a scalar.
// =============================================================================
template <typename T> __global__ void
kernelMulScalar ( /* argument list: {{{*/
    T* result, T c, T* x, size_t size 
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* kernelMulScalar implementation: {{{*/
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  size_t gridSize = blockDim.x * gridDim.x;
  
  while (tid < size) {
    result[tid] = c * x[tid];
    tid += gridSize;
  }
} /*}}}*/
/* ---------------  end of function kernelMulScalar  -------------------- }}}*/

// === kernelAdd: TEMPLATE FUNCTION  =========================================={{{ 
//         Name:  kernelAdd
//  Description:  The sum of two vectors.
// =============================================================================
template <typename T> 
__global__ void
kernelAdd ( /* argument list: {{{*/
    T* result, T* lhs, T* rhs, size_t size 
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /*  implementation: {{{*/
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  size_t gridSize = blockDim.x * gridDim.x;
  
  while (tid < size) {
    result[tid] = lhs[tid] + rhs[tid];
    tid += gridSize;
  }
} /*}}}*/
/* ---------------  end of function kernelAdd  -------------------- }}}*/

// === kernelPlusMinus: TEMPLATE FUNCTION  =========================================={{{ 
//         Name:  kernelPlusMinus
//  Description:  Return the positive part and negative part of a vector.
// =============================================================================
template <typename T> __global__ void
kernelPlusMinus ( /* argument list: {{{*/
    T* plus, T* minus, T* data, size_t size 
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /*  implementation: {{{*/
  size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  size_t gridSize = blockDim.x * gridDim.x;

  while (tid < size) {
    plus[tid] = 0;
    minus[tid] = 0;
    if (data[tid] > 0) 
      plus[tid] = data[tid];
    else
      minus[tid] = data[tid];
    tid += gridSize;
  }
} /*}}}*/
/* ---------------  end of function kernelPlusMinus  -------------------- }}}*/

// === kernelAbs: TEMPLATE FUNCTION  =========================================={{{ 
//         Name:  kernelAbs
//  Description:  Return the absolute value of a vector element wise.
// =============================================================================
template <typename T> 
__global__ void
kernelAbs ( /* argument list: {{{*/
    T* a, size_t size 
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /*  implementation: {{{*/
  uint tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint gridSize = gridDim.x * blockDim.x;
  while (tid < size) {
    a[tid] = abs(a[tid]);
    tid += gridSize;
  }
} /*}}}*/
/* ---------------  end of function kernelAbs  -------------------- }}}*/

template <> void 
zMalloc <HOST> (void** ptr, size_t size) {/*{{{*/
  if (size != 0) {
    *ptr	=  malloc ( size );
    if ( *ptr == NULL ) {
      printf("Dynamic memory failed in %s at line %d\n", __FILE__, __LINE__);
      exit (EXIT_FAILURE);
    }
  }
  else {
    *ptr = NULL;
  }
//  hostMallocCount++;
}/*}}}*/

template <> void 
zMalloc <GPU> (void** ptr, size_t size) {/*{{{*/
  if (size != 0)
    HANDLE_ERROR (cudaMalloc (ptr, size));
  else
    *ptr = NULL;
//  cudaMallocCount++;
}/*}}}*/

template <> void 
zFree <HOST> (void* ptr) {/*{{{*/
//  if (ptr == NULL) hostFreeCount--;
  free(ptr);
  ptr = NULL;
//  hostFreeCount++;
}/*}}}*/

template <> void 
zFree <GPU> (void* ptr) {/*{{{*/
//  if (ptr == NULL) cudaFreeCount--;
  HANDLE_ERROR (cudaFree (ptr));
  ptr = NULL;
//  cudaFreeCount++;
}/*}}}*/


template <> void
zMemcpy <HOST> (void* to, void* from, size_t size) {/*{{{*/
  memcpy ( to, from, size );
}/*}}}*/

template <> void
zMemcpy <GPU> (void* to, void* from, size_t size) {/*{{{*/
  uint dimGrid, dimBlock;
  kernelConfig (dimGrid, dimBlock, size);
  kernelMemcpy <<< dimGrid, dimBlock >>> (to, from, size);
}/*}}}*/

template <deviceType toDevice, deviceType fromDevice> void 
zMemcpy (void* to, void* from, size_t size) {/*{{{*/
  if ((toDevice == GPU) && (fromDevice == GPU)) {
    zMemcpy <GPU> (to, from, size);
  }
  if ((toDevice == GPU) && (fromDevice == HOST)) {
    HANDLE_ERROR (cudaMemcpy (to, from, size, cudaMemcpyHostToDevice));
  }
  if ((toDevice == HOST) && (fromDevice == GPU)) {
    HANDLE_ERROR (cudaMemcpy (to, from, size, cudaMemcpyDeviceToHost));
  }
  if ((toDevice == HOST) && (fromDevice == HOST)) {
    zMemcpy <HOST> (to, from, size);
  }
}/*}}}*/


// === kernelMemset: CUDA KERNEL ========================================{{{
//         Name:  kernelMemset
//  Description:  Set all elements of a matrix to a fix value.
// =============================================================================
__global__ void
kernelMemset ( /* argument list: {{{*/
    double* ptr, double value, size_t size 
    ) /* ------------- end of argument list ------------------------------}}}*/ 
{ /* kernelMemset implementation: {{{*/
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t gridSize = blockDim.x * gridDim.x;
  while (tid < size) {
    ptr[tid] = value;
    tid += gridSize;
  }
} /*}}}*/
/* ----------------  end of CUDA kernel kernelMemset  ----------------- }}}*/

template <> void
zMemset <HOST> (double* ptr, double value, size_t size) {/*{{{*/
  assert (ptr != NULL);
  assert (size > 0);
  for ( size_t i = 0; i < size; i++ ) { /*  {{{*/
    ptr[i] = value;
  }         /*---------- end of for loop ----------------}}}*/
}/*}}}*/

// === kernelConfig: FUNCTION  =========================================={{{ 
//         Name:  kernelConfig
//  Description:  Compute the 'optimize' dimGrid and dimBlock.
// =============================================================================
void
kernelConfig ( /* argument list: {{{*/
    uint & dimGrid, uint & dimBlock, const size_t & size 
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* kernelConfig implementation: {{{*/
  dimBlock = chooseSize (size);
  if (dimBlock > 512) dimBlock = 512;
  dimGrid  = chooseSize (size / dimBlock);
  if (dimGrid == 0) dimGrid = 1;
} /*}}}*/
/* ---------------  end of function kernelConfig  -------------------- }}}*/

template <> void
zMemset <GPU> (double* ptr, double value, size_t size) {/*{{{*/
  assert (ptr != NULL);
  assert (size > 0);
  uint dimBlock, dimGrid;
  kernelConfig (dimGrid, dimBlock, size);
  kernelMemset <<<dimGrid, dimBlock>>> (ptr, value, size);
}/*}}}*/

// === hostMax: TEMPLATE FUNCTION  =========================================={{{ 
//         Name:  hostMax
//  Description:  Return the max value of a vector of size <= 1024 using
//  reduction on host.
// =============================================================================
template <typename T, deviceType dataLocation> T
hostMax ( /* argument list: {{{*/
    T* data, size_t size 
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /*  implementation: {{{*/

  if (dataLocation == HOST) {
    Eigen::Map <Eigen::Matrix <T, Eigen::Dynamic, Eigen::Dynamic> > 
      edata (data, size, 1);
    return edata.maxCoeff();
  } 
  else {
    T* b;
    T retval;
    zMalloc <HOST> ((void**)&b, size * sizeof(T));
    zMemcpy <HOST, dataLocation> ((void*)b, (void*)data, size * sizeof(T));
//  size_t k;
//  while (size > 1) {
//    k = 0;
//    while (k < size - k - 1) {
//      if (b[k] < b[size - k - 1]) b[k] = b[size - k - 1];
//      k++;
//    }
//    size = (size + 1)/2;
//  }
    Eigen::Map <Eigen::Matrix <T, Eigen::Dynamic, Eigen::Dynamic> > 
    eb (b, size, 1);
    retval = eb.maxCoeff();
    zFree <HOST> (b);
    return retval;
  }
} /*}}}*/
/* ---------------  end of function hostMax  -------------------- }}}*/

template < typename T > T
exactSum ( T* data, size_t size) {/*{{{*/
  T* partials;
  T hi, lo, tx, ty;
  size_t sizey = 0;
  size_t k;
  zMalloc <HOST> ((void**)&partials, size * sizeof(T));
  
  for ( size_t i = 0; i < size; i++ ) { /* loop on the original data {{{*/
    tx = data[i];
    k = 0;
    for ( size_t j = 0; j < sizey; j++ ) { /* loop on partials sum {{{*/
      ty = partials[j];
      if (abs(tx) < abs(ty)) swap (tx, ty);
      hi = tx + ty;
      lo = ty - (hi - tx);
      if (lo) {
        partials[k] = lo;
        k++;
      }
      tx = hi;
    }         /*---------- end of for loop ----------------}}}*/
    partials[k] = tx;
    k++;
    sizey = k;
  }         /*---------- end of for loop ----------------}}}*/

  hi = 0;
  for ( size_t i = 0; i < sizey; i++ ) { /* final sum {{{*/
    hi += partials[i];
  }         /*---------- end of for loop ----------------}}}*/
  return hi;
}/*}}}*/

// === warpReduce: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  warpReduce
//  Description:  Sum reduction within a warp.
// =============================================================================
__device__ __host__ void
warpReduce ( /* argument list: {{{*/
    volatile double* data, const int& tid, const int& blocksize 
    ) /* ------------- end of argument list -----------------------------------}}}*/ 
{ /* warpreduce implementation: {{{*/
  if ( blocksize > 32 ) if ( tid < 32 ) data [ tid ] += data [ tid + 32 ] ;
  if ( blocksize > 16 ) if ( tid < 16 ) data [ tid ] += data [ tid + 16 ] ;
  if ( blocksize >  8 ) if ( tid <  8 ) data [ tid ] += data [ tid +  8 ] ;
  if ( blocksize >  4 ) if ( tid <  4 ) data [ tid ] += data [ tid +  4 ] ;
  if ( blocksize >  2 ) if ( tid <  2 ) data [ tid ] += data [ tid +  2 ] ;
  if ( blocksize >  1 ) if ( tid <  1 ) data [ tid ] += data [ tid +  1 ] ;
} /*}}}*/
/* ---------------  end of DEVICE function warpReduce  -------------- }}}*/

// === readData: FUNCTION  =========================================={{{ 
//         Name:  readData
//  Description:  Load parameters from a text file. The file is structured into
//  two columns separated by a semicolon and zero or more spaces. You can also
//  choose your own delimiters. The left
//  column contains the names of the parameters and the second column contains
//  their numerical values.
// =============================================================================
int
readData ( /* argument list: {{{*/
    //double* values, const std::string names[], const int& numel
    std::map< std::string, double > & params
    , const char *fileName
    , const char *delimiters = ": "
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* readData implementation: {{{*/
  std::ifstream param(fileName);
  if (!param.good()) {
    std::cout << "Cannot open file!" << std::endl;
    return 1;
  }
  
  char buf[80];
  char *key;
  char *value;

  while (!param.eof()) {
    param.getline(buf, 80);
    key = strtok(buf, delimiters);
    if (key) {
      value = strtok(NULL, delimiters);
      params[key] = atof(value);
    }
  }
  param.close();
  return 0;
} /*}}}*/
/* ---------------  end of function readData  -------------------- }}}*/

#endif   // ----- #ifndef MYSTDLIB_H_  ----- 
