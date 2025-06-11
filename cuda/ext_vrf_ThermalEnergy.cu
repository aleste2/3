#include <stdint.h>
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "stdio.h"

extern "C" __global__ void
addthermalenergydensity(float* __restrict__ energy, 
                 float* __restrict__ Te, float* __restrict__ Tl,
                 float* __restrict__ ce_, float ce_mul,
                 float* __restrict__ cl_, float cl_mul,
                 int N) {

    int I =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (I < N) {
        float Ce = amul(ce_, ce_mul, I);
        float Cl= amul(cl_, cl_mul, I);
        energy[I] = Te[I]*Ce*Te[I]/300.0+Tl[I]*Cl;
    }
}