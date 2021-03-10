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

// Langevin function
inline __device__ float dL(float x)
{

  double temp = exp(2*x)+(double)1.0;
  if(fabs(temp)>1e308) return x>0 ? 1.0 : -1.0; // Large input
  temp /= temp-2; // temp = coth(x);
  //return 0.0001f+(temp-1/x)/1.05;
  double value=temp-(double)1.0/x;
  if (value<0.03) value=0.03;
  
  return value;
}

// Langevin derivative function (de verdad)
inline __device__ float dLder(float x)
{
	double value=1.0f/(x*x)-pow(1.0f/sinh(x),2.0f);
  	if (value<0.03) value=0.03;	
	return (value);
}

// Langevin function
inline __device__ float dL0(float x)
{

  double temp = exp(2*x)+(double)1.0;
  if(fabs(temp)>1e308) return x>0 ? 1.0 : -1.0; // Large input
  temp /= temp-2; // temp = coth(x);
  //return 0.0001f+(temp-1/x)/1.05;
  double value=temp-(double)1.0/x;
 
  return value;
}

// Langevin derivative function (de verdad)
inline __device__ float dLder0(float x)
{
	double value=1.0f/(x*x)-pow(1.0f/sinh(x),2.0f);	
	return (value);
}



