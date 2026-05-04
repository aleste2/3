//#include <stdint.h>
#include "amul.h"
#include "float3.h"
//#include "stencil.h"
#include "constants.h"

extern "C"

__global__ void
museffectivefield(
    float* __restrict__ dstx, float* __restrict__ dsty, float* __restrict__ dstz,
    float* __restrict__ musx, float* __restrict__ musy, float* __restrict__ musz,
    float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
    float* __restrict__ Tc_, float Tc_mul,
    float* __restrict__ Ms_, float Ms_mul,
    int N
) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
     float3 m0={mx[i], my[i], mz[i]};
     float mm=dot(m0,m0);
     if (mm!=0) {
        float kb=1.380649e-23;
        float Tc = amul(Tc_, Tc_mul, i);
        float Ms = amul(Ms_, Ms_mul, i);
        float prefactor= -1.0f*MU0*Ms/(Tc*2.0f*kb);
        //if (i==1) {printf("%e %e\n",prefactor,musz[i]);}
        //printf("%e %e\n",prefactor,musz[i]);
        dstx[i]+=prefactor*musx[i];
        dsty[i]+=prefactor*musy[i];
        dstz[i]+=prefactor*musz[i];
     }
    }
}
