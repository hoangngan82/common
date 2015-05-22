// =====================================================================================
//
//       Filename:  sm.h
//        Version:  1.0
//        Created:  05/20/2015 11:07:12 PM
//       Revision:  none
//       Compiler:  nvcc, g++
//      Copyright:  Copyright (c) 2015, Hoang-Ngan Nguyen
//          Email:  zhoangngan [@] gmail
//        License:  MIT License
//
//    Description:  A data type tha stores double values but gets around the
//    underflow and overflow behaviors.
//
// =====================================================================================

#ifndef SIMPLE_MATH_H
#define SIMPLE_MATH_H

#include <cmath>
#include <iostream>
using std::cout;
using std::endl;

#ifdef __cplusplus
namespace sm {
#endif

const unsigned short MSW = 3; /* most significant word */

typedef union {
  double          d;
  unsigned short  s[4];
  unsigned long   l;
} ieee754;

/* multiplied by power of 2: y = (x)*(2^n) */
double mul_pow2( const double & x, const int & n ) 
{
  ieee754 y;
  y.d = (x < 0) ? (-x) : x;
  if( y.d == 0 ) return y.d;

  /* denormalized numbers */
  long m = n + (long)(y.l >> 52);
  if( m <= 0 ) {
    m = 1-m;
    y.s[MSW] &= 0x000F;
    y.s[MSW] |= 0x0010;

    y.l >>= m;

    if( x < 0 ) y.s[MSW] |= 0x8000;
    return y.d;
  }

  /* infinity */
  if( m > 2046 ) {
    y.l = 0;
    if( x < 0 ) y.s[MSW] = 0xFFF0; else y.s[MSW] = 0x7FF0;
    return y.d;
  }

  /* normalized number */
  y.d = x;
  m = (long)n << 52;
  m += (long)y.l;
  y.l = (unsigned long)m;
  return y.d;
}

class Real {
  public:
    /* data */
    double frac;
    int    expo;

    /* default constructors */
    Real( const double & x = 0 ) { frac = frexp( x, &expo ); }

    Real operator*( const Real & rhs ) const {
      Real result;
      result.frac = frac * rhs.frac;
      result.frac = frexp( result.frac, &result.expo );
      result.expo += expo + rhs.expo;
      return result;
    }
    Real operator/( const Real & rhs ) const {
      Real result;
      result.frac = frac / rhs.frac;
      result.frac = frexp( result.frac, &result.expo );
      result.expo += expo - rhs.expo;
      return result;
    }
    Real operator+( const Real & rhs ) const {
      Real result;
      int  dx = expo - rhs.expo;
      if( dx <= 0 ) {
        result.frac = mul_pow2(frac, dx) + rhs.frac;
        result.frac = frexp( result.frac, &result.expo );
        result.expo+= rhs.expo;
      } else {
        result.frac = frac + mul_pow2(rhs.frac, -dx);
        result.frac = frexp( result.frac, &result.expo );
        result.expo+= expo;
      }
      return result;
    }
    Real operator-( const Real & rhs ) const {
      Real result;
      if( *this == rhs ) {
        result.frac = 0;
        result.expo = 0;
        return result;
      }
      int  dx = expo - rhs.expo;
      if( dx <= 0 ) {
        result.frac = mul_pow2(frac, dx) - rhs.frac;
        result.frac = frexp( result.frac, &result.expo );
        result.expo+= rhs.expo;
      } else {
        result.frac = frac - mul_pow2(rhs.frac, -dx);
        result.frac = frexp( result.frac, &result.expo );
        result.expo+= expo;
      }
      return result;
    }

    double eval() const { 
      return mul_pow2(frac, expo);
      //ieee754 m; 
      //m.d = frac;
      //long n = expo;
      //if( (n > -1022) && (n < 1023) ) {
        //n <<= 52;
        //n += (long)(m.l);
        //m.l = (unsigned long)(n);
        //return m.d;
      //}
      //else return (frac*::pow(2, expo));
    }

    bool operator==( const Real & rhs ) const {
      if( (frac == rhs.frac ) && (expo == rhs.expo) ) 
        return true;
      else
        return false;
    }

    friend std::ostream & operator<<(std::ostream & os, const Real & x) {
      os << "frac = " << x.frac << ", expo = " << x.expo 
        << ", and val = " << x.eval() << endl;
      return os;
    }
};

Real frac( const Real & x ) {

  const unsigned long twop12 = ~((0xFFFUL << 52));

  ieee754 m;
  m.d = x.frac;
  m.l &= 0x7FFFFFFFFFFFFFFF;

  unsigned long ufrac = m.l & twop12;
  long ex = x.expo;
  if( ex <= 0 ) if( x.frac >= 0 ) return x; else return (Real(1.0) - x);
  if( ex > 52 ) return Real(0.0);
  ex--;
  ufrac <<= ex;

  unsigned long i = 1UL << 51;
  ex = 1;
  while( ((ufrac & i) == 0) && (i != 0) ) {
    i >>= 1;
    ex++;
  }

  ufrac <<= ex;
  ufrac &= twop12;
  ex  = 1023 - ex;
  ex  = (long)ufrac + (ex << 52);
  m.l = (unsigned long)ex;
  if( x.frac >= 0 )
    return Real(m.d);
  else
    return Real(1.0 - m.d);
}

int floor( const Real & x ) {
  const unsigned long twop12 = ~((0xFFFUL << 52));

  ieee754 m;
  m.d = x.frac;
  //if( m.d < 0 ) m.d = -m.d;
  m.l &= 0x7FFFFFFFFFFFFFFF;

  unsigned long ufrac = m.l & twop12;
  ufrac = ufrac | (twop12 + 1);

  long ex = m.l >> 52;
  ex = ex + x.expo - 1023;

  int retval = 0;
  if( ex >= 52 ) {
    retval = ufrac*(1 << (ex - 52));
  } else 
    if( ex >= 0 ) {
    retval = (ufrac >> (52 - ex));
  }
  if( x.frac < 0 ) retval = -retval - 1;
  return retval;
}

Real pow( const Real & x, const int & n ) {
  const int step = 1000;

  if( (x == Real(0)) && (n <= 0) ) {
    cout << "Undefined value 0^n with n <= 0!!! Returning 0!!!" << endl;
    return x;
  }
  if( x == Real(0) ) return x;
  if( x == Real(1) ) return x;
  if( x == Real(-1) ) if( (n & 1) == 0 ) return x; else return Real(-1);
  if( n == 0 ) return Real(1);
  Real result(1.0);
  int m = n;
  if( n < 0 ) m = -n;
  if( m <= step ) {
    result.frac = ::pow(x.frac, n);
    result.frac = frexp(result.frac, &result.expo);
    result.expo+= x.expo*n;
    return result;
  }

  /* |n| > 1000 */
  int nn = step;
  Real temp;
  if( n < 0 ) nn = -nn;
  temp.frac = ::pow(x.frac, nn);
  temp.frac = frexp(temp.frac, &temp.expo);
  temp.expo+= x.expo*nn;

  while( m > step ) {
    result = result*temp;
    m -= step;
  }

  /* the remaining part */
  if( n < 0 ) m = -m;
  temp.frac = ::pow(x.frac, m);
  temp.frac = frexp(temp.frac, &temp.expo);
  temp.expo+= x.expo*m;
  result = result*temp;
  return result;
}

//// A WRONG way to do pow!!!
//Real pow( const Real & x, const int & n ) {
  //if( (x == Real(0)) && (n <= 0) ) {
    //cout << "Undefined value 0^n with n <= 0!!! Returning 0!!!" << endl;
    //return x;
  //}
  //if( x == Real(0) ) return x;
  //if( x == Real(1) ) return x;
  //if( x == Real(-1) ) if( (n & 1) == 0 ) return x; else return Real(-1);
  //if( n == 0 ) return Real(1);
  //Real result;
  //int m = n;
  ////if( n < 0 ) m = -n;
  //result.frac = pow(x.frac, m);
  //result.frac = frexp(result.frac, &result.expo);
  //result.expo+= x.expo*m;
  ////if( n < 0 ) result = Real(1.0)/result;
  //return result;
//}

Real exp( const Real & x ) {
  if( x.expo < -51 ) return Real(1.0);
  Real retval;
  int y = x.expo;
  if( y < 10 ) return Real( ::exp( x.eval() ) );
  retval.frac = x.frac;
  retval.expo = 9;
  retval = Real( ::exp( retval.eval() ) );
  return pow( retval, (1UL << (y - 9)) );
}

#ifdef __cplusplus
} /* end of name space sm */
#endif

#endif /* end of SIMPLE_MATH_H */
