#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"
#include "exchange.h"

extern "C"

__global__ void
addMusTorqueAtomu(
    float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
    float* __restrict__ musx, float* __restrict__ musy, float* __restrict__ musz,
    float* __restrict__ mu_, float mu_mul,
    float* __restrict__ pol_, float pol_mul,
    float* __restrict__ epsprime_, float epsprime_mul,
    float* __restrict__ alpha_, float alpha_mul,
    float dt,
    int N
) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
        float3 m={mx[i], my[i], mz[i]};
        float3 mus={musx[i], musy[i], musz[i]};
        float mm=dot(m,m);
        if (mm!=0) {
            float mu = amul(mu_, mu_mul, i);
            float pol = amul(pol_, pol_mul, i);
            float epsprime = amul(epsprime_, epsprime_mul, i);
            float alpha = amul(alpha_, alpha_mul, i);
            float prefactor= dt/mu;
            float premuxm=(epsprime-pol*alpha)/(1.0f+alpha*alpha);
            float premxmuxm=(pol+epsprime*alpha)/(1.0f+alpha*alpha);
            //float3 torque=cross(mus,m)+cross(m,cross(mus,m));
            float3 torque=((premuxm/sqrt(mm))*cross(mus,m))+((premxmuxm/mm)*cross(m,cross(mus,m)));  // Scaled for LLB if needed
            mx[i]+=prefactor*torque.x;
            my[i]+=prefactor*torque.y;
            mz[i]+=prefactor*torque.z;
        }
    }
}