#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"
#include "exchange.h"

extern "C"

__global__ void
calcTorqueJs(
    float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
    float* __restrict__ Jsxx, float* __restrict__ Jsxy, float* __restrict__ Jsxz,
    float* __restrict__ Jsyx, float* __restrict__ Jsyy, float* __restrict__ Jsyz,
    float* __restrict__ Jszx, float* __restrict__ Jszy, float* __restrict__ Jszz,
    float* __restrict__ mu_, float mu_mul,
    float scale,
    float dt,
    int N
) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
        float3 m={mx[i], my[i], mz[i]};
        float3 Jsx={Jsxx[i], Jsxy[i], Jsxz[i]};
        float3 Jsy={Jsyx[i], Jsyy[i], Jsyz[i]};
        float3 Jsz={Jszx[i], Jszy[i], Jszz[i]};
        float mm=dot(m,m);
        if (mm!=0) {
            float3 ux={1, 0, 0};
            float3 uy={0, 1, 0};
            float3 uz={0, 0, 1};

            float mu = amul(mu_, mu_mul, i);
            float prefactor= dt*scale/mu;

            float T0x=prefactor*(Jsx.x+Jsx.y+Jsx.y);
            float T0y=prefactor*(Jsy.x+Jsy.y+Jsy.y);
            float T0z=prefactor*(Jsz.x+Jsz.y+Jsz.y);

            float3 torque=T0x*cross(m,cross(ux,m))+T0y*cross(m,cross(uy,m))+T0z*cross(m,cross(uz,m));
            mx[i]+=torque.x;
            my[i]+=torque.y;
            mz[i]+=torque.z;
        }
    }

}