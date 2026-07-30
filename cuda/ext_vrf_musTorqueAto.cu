#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"
#include "exchange.h"

extern "C"

__global__ void
addMusTorqueAto(
    float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
    float* __restrict__ Jsx, float* __restrict__ Jsy, float* __restrict__ Jsz,
    float* __restrict__ mu_, float mu_mul,
    float scale,
    float dt,
    int N
) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
        float3 m={mx[i], my[i], mz[i]};
        float3 Js={Jsx[i], Jsy[i], Jsz[i]};
        float mm=dot(m,m);
        if (mm!=0) {
            float mu = amul(mu_, mu_mul, i);
            float prefactor= dt*scale/mu;
            float3 torque=cross(m,cross(Js,m));
            mx[i]+=prefactor*torque.x;
            my[i]+=prefactor*torque.y;
            mz[i]+=prefactor*torque.z;
        }
    }
}