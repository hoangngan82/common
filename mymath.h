
/*
 * =====================================================================================
 *
 *       Filename:  mymath.h
 *
 *    Description:  contains wrappers for other library
 *
 *        Version:  1.0
 *        Created:  04/12/2015 10:25:28 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Hoang-Ngan Nguyen (), zhoangngan-gmail
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef MY_MATH_H__
#define MY_MATH_H__

#include "mystdlib.h"
#include <cuda.h>
#include <math_functions.h>
#include <fftw3.h>

#ifndef PI
#  define PI   (+3.14159265358979324)
#  define SPI  (+1.77245385090551603)
#  define eps  (+2.22044604925031e-16)
#endif

// === shiftedFFT:  making the wave number centered at 0. ===={{{
// =============================================================================
void
shiftedFFT ( /* argument list: {{{*/
    fftw_complex *in
  , fftw_complex *out
  , const uint & nx       // number of elements in x-direction
  , const uint & ny = 1   // number of elements in y-direction
  , const uint & nz = 1   // number of elements in z-direction
  , const int  & forward = FFTW_FORWARD
    ) /* ------------- end of argument list -----------------------------}}}*/ 
{ /* shiftedFFT implementation: {{{*/
  uint mx = ( nx + 1 ) >> 1;
  uint my = ( ny + 1 ) >> 1;
  uint mz = ( nz + 1 ) >> 1;

  if( forward == FFTW_FORWARD )     // symmetric wave numbers
    for( int i = 0; i < nx; i++ ) {
      uint im = ( i + mx )%nx;
      for( int j = 0; j < ny; j++ ) {
        uint jm = ( j + my )%ny;
        for( int k = 0; k < nz; k++ ) {
          uint km = ( k + mz )%nz;
          out[ ( i*ny + j )*nz + k ][0] = in[ ( im*ny + jm )*nz + km ][0];
          out[ ( i*ny + j )*nz + k ][1] = in[ ( im*ny + jm )*nz + km ][1];
        }
      }
    }
  else              // non-negative wave numbers
    for( int i = 0; i < nx; i++ ) {
      uint im = ( i + mx )%nx;
      for( int j = 0; j < ny; j++ ) {
        uint jm = ( j + my )%ny;
        for( int k = 0; k < nz; k++ ) {
          uint km = ( k + mz )%nz;
          out[ ( im*ny + jm )*nz + km ][0] = in[ ( i*ny + j )*nz + k ][0];
          out[ ( im*ny + jm )*nz + km ][1] = in[ ( i*ny + j )*nz + k ][1];
        }
      }
    }
} /*}}}*/
/* ---------------  end of function shiftedFFT  -------------------- }}}*/

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

// === experf: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name:  experf
//  Description: Compute exp( c*x*z )*erfc( z + x ); 
// =============================================================================
__device__ __host__ double 
experf (double x, double z, const double &c = 2) {
  double a = 0;
  if ((x*z < 0) || (x*x + z*z < 700))
    a = exp(c*x*z) * erfc(z + x);
  return a;
}
/* ---------------  end of DEVICE function experf  -------------- }}}*/

// === cross: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name: cross 
//  Description: Compute the cross product of two vectors.
// =============================================================================
template <typename T> 
__device__ __host__ void
cross (T* result, T* a, T* b) {
  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];
}
/* ---------------  end of DEVICE function cross  -------------- }}}*/

// === exactSum: CUDA DEVICE AND HOST FUNCTION ======================{{{
//         Name: exactSum
//  Description: Compute the exact sum of a sequence of real numbers without
//  loosing accuracy due to cancellation errors.
// =============================================================================
template < typename T > 
__device__ __host__ T
exactSum ( T* data, size_t size ) 
{
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
}
/* ---------------  end of DEVICE function cross  -------------- }}}*/

#endif // MY_MATH_H__

