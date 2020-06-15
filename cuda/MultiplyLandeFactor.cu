#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// See exchange.go for more details.
extern "C" __global__ void
MultiplyLandeFactor(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
            float* __restrict__ g_, float g_mul,
            int N
            ) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float  g  = amul(g_, g_mul, i)/2.0f;
        if (g==0) g=1.0f;
        Bx[i] = g*Bx[i];
        By[i] = g*By[i];
        Bz[i] = g*Bz[i];
    }
}