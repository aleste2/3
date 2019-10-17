#include "float3.h"
#include "math.h"

// Langevin function
inline __device__ float Brillouin(float x,float J)
{
  return ( (2.0f*J+1.0f)/(2.0f*J)/tanh((2.0f*J+1.0f)/(2.0f*J)*x) - 1.0f/(2.0f*J)/tanh(x/(2.0f*J)));
}

// Langevin derivative function
inline __device__ float DBrillouin(float x,float J)
{
 // if (x<6){
  return ( pow(sinh(x/(2.0f*J)),-2.0f) /(4.0f*J*J) - (1.0f+2.0f*J)*(1.0f+2.0f*J)* pow(sinh((1.0f+2.0f*J)/(2.0f*J)*x),-2.0f) /(4.0f*J*J));
 // }else{
 // return(0.000025073);
 // }
}

// Langevin derivative function (de verdad)
inline __device__ float Lder(float x)
{
	return (1.0f/(x*x)-pow(1.0f/sinh(x),2.0f));
}

