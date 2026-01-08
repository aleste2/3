#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
ElasticEnergy(float* __restrict__ energy, 
                 float* __restrict__ exx, float* __restrict__ eyy, float* __restrict__ exy,
                 float* __restrict__  C1_, float  C1_mul, float* __restrict__  C2_, float  C2_mul, 
                 float* __restrict__  C3_, float  C3_mul,
                 int N) {

    int I =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (I < N) {

    // Central cell
    float c1 = amul(C1_, C1_mul, I);
    float c2 = amul(C2_, C2_mul, I);
    float c3 = amul(C3_, C3_mul, I);

    energy[I] = c1*0.5*(exx[I]*exx[I]+eyy[I]*eyy[I]);
    energy[I] += c2*(exx[I]*eyy[I]);
    energy[I] += c3*2*(exy[I]*exy[I]);
    }
}