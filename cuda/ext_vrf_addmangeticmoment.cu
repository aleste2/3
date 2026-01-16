#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"
#include "exchange.h"

extern "C"

__global__ void
addmagneticmoment(
    float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
    float* __restrict__ musx, float* __restrict__ musy, float* __restrict__ musz,
    float* __restrict__ Tc_, float Tc_mul,
    float* __restrict__ Tausd_, float Tausd_mul,
    float dt,
    int N
) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
     float3 m0={mx[i], my[i], mz[i]};
     float mm=dot(m0,m0);
     if (mm!=0) {
        float kb=1.380649e-23;
        float Tc = amul(Tc_, Tc_mul, i);
        float tausd = amul(Tausd_, Tausd_mul, i);
        float prefactor= dt/(tausd*Tc*2.0f*kb);
        mx[i]-=prefactor*musx[i];
        my[i]-=prefactor*musy[i];
        mz[i]-=prefactor*musz[i];
     }
    }
}
