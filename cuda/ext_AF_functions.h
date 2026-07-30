#ifndef _ext_AF_functions_H_
#define _ext_AF_functions_H_

#include "float3.h"
#include "math.h"

// Langevin function
inline __device__ float Langevin(float x)
{
  float temp = exp(2*x)+1;
  //if(fabs(temp)>1e200) return x>0 ? 1.0 : -1.0; // Large input
    if(fabs(temp)>3.4e38) return x>0 ? 1.0 : -1.0; // Large input
  temp /= temp-2; // temp = coth(x);
  return temp-1/x;
}


// Langevin derivative function
inline __device__ float LangevinDeriv(float x)
{
  //if(fabs(x)<1e-30) return 1./3.;
  if(fabs(x)<1.18e-38) return 1./3.;
  float temp = sinh(x);
  //if(fabs(temp)>1e200) return 1.0/(x*x); // Large input
if(fabs(temp)>3.4e38) return 1.0/(x*x); // Large input
  return -1.0/(temp*temp)+1.0/(x*x);
}



// J0nu_gorro Implementacion segun eq (15) PRB 86 104414 (2012)
inline __device__ float J0_(float J0, float J0int,float me1,float me2)
{
  return (J0*me1+fabs(J0int)*me2)/me1;
}

// Bbu  Implementacion segun eq (25) PRB 86 104414 (2012)
inline __device__ float Bnu(float T,float3 m1,float3 m2,float J0,float J0int)
{
  float m1m=pow(dot(m1,m1),0.5);
  float m2m=pow(dot(m2,m2),0.5);
  float tauk=dot(m1,m2)/dot(m1,m1);
  return Langevin(J0*m1m+fabs(J0int)*tauk);
}

// Alpha paraelell Implementacion segun eq (14) PRB 86 104414 (2012)
inline __device__ float alphapar(float T,float me1,float me2,float lambda,float J0,float J0int)
{
  float kB= 1.38064852e-23;
  return 2.0f*lambda/( J0_(J0,J0int,me1,me2)/(kB*T) );
}

// Alpha perpendicular Implementacion segun eq (14) PRB 86 104414 (2012)
inline __device__ float alphaperp(float T,float me1,float me2,float lambda,float J0,float J0int)
{
  float kB= 1.38064852e-23;
  return lambda*( 1.0f-1.0f/( J0_(J0,J0int,me1,me2)/(kB*T)) );
}

// Test LLBtorqueAF2TBrillouin

inline __device__ float Brillouin2(float x, float J)
{
  if (fabs(x)<0.01) {return(x);} else {
  return ( (2.0f*J+1.0f)/(2.0f*J)/tanh((2.0f*J+1.0f)/(2.0f*J)*x) - 1.0f/(2.0f*J)/tanh(x/(2.0f*J)));}

/*
  float temp = exp(2*x)+1;
  //if(fabs(temp)>1e200) return x>0 ? 1.0 : -1.0; // Large input
    if(fabs(temp)>3.4e38) return x>0 ? 1.0 : -1.0; // Large input
  temp /= temp-2; // temp = coth(x);
  return temp-1/x;
*/

/*
//New Langevin
 if (x==0) {return(0.0f);} else {return(1.0f/tanh(x)-1/x);}
*/
}

inline __device__ float DBrillouin2(float x, float J)
{
  if (fabs(x)<0.01) {return(1.0f);} else {
    return ( pow(sinh(x/(2.0f*J)),-2.0f) /(4.0f*J*J) - (1.0f+2.0f*J)*(1.0f+2.0f*J)* pow(sinh((1.0f+2.0f*J)/(2.0f*J)*x),-2.0f) /(4.0f*J*J));}

// New langevin
/*
if (fabs(x)<0.01) {
  return(1.0f);} else {
    return (-pow(sinh(x),-2.0f) + 1.0f/(x*x) );}
*/
}
#endif
