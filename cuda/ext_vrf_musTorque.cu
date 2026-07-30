#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"
#include "exchange.h"

extern "C"

__global__ void
addMusTorque(
    float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
    float* __restrict__ musx, float* __restrict__ musy, float* __restrict__ musz,
    float* __restrict__ Ms_, float Ms_mul,
    float* __restrict__ Sigma_, float Sigma_mul,
    float* __restrict__ LambdaJ_, float LambdaJ_mul,
    float* __restrict__ Betta_, float Betta_mul,
    float dt,
    int N
) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
     float3 m={mx[i], my[i], mz[i]};
     float3 mu={musx[i], musy[i], musz[i]};
     float mm=dot(m,m);
     if (mm!=0) {
        float Msat = amul(Ms_, Ms_mul, i);
        float Sigma = amul(Sigma_, Sigma_mul, i);
        float LJ = amul(LambdaJ_, LambdaJ_mul, i);
        float LJ2=LJ*LJ;
        float betta = amul(Betta_, Betta_mul, i);
        float GammaLL= 1.7595e11;
        //float qe=1.60217646E-19;
        float mub=9.27e-24;
        //float hbar=1.05457173E-34;
        float prefactor= GammaLL/mub;
        float3 torque=cross(m,mu)+betta*cross(m,cross(m,mu));
        //if (i==10) printf("%e\n",prefactor);
        if ((LJ!=0)&&(Msat!=0)){
          //  prefactor=100e34;
          prefactor=prefactor*dt;
          //if (i==10) printf("%e %e  %e %e\n",prefactor,prefactor*torque.x,prefactor*torque.y,prefactor*torque.z);
          mx[i]+=prefactor*torque.x;
          my[i]+=prefactor*torque.y;
          mz[i]+=prefactor*torque.z;
        }

     }
    }
}