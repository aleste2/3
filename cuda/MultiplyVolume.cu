#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// See exchange.go for more details.
extern "C" __global__ void
MultiplyVolume(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
            float volume,
            int N
            ) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
        Bx[i] = volume*Bx[i];
        By[i] = volume*By[i];
        Bz[i] = volume*Bz[i];
    }
}