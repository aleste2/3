#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
KineticEnergy(float* __restrict__ energy, 
                 float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
                 float* __restrict__ rho,
                 int N) {

    int I =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (I < N) {
        energy[I] = 0.5* rho[I]* (vx[I]*vx[I]+vy[I]*vy[I]+vz[I]*vz[I]);
    }
}