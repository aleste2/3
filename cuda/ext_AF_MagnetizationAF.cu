#include "amul.h"
#include "float3.h"
#include <stdint.h>

// normalize m: m=(m1*Ms1+m2*Ms2)/Ms0
extern "C" __global__ void
MagnetizationAF(float* __restrict__ m0x, float* __restrict__ m0y, float* __restrict__ m0z,
	float* __restrict__ m1x, float* __restrict__ m1y, float* __restrict__ m1z,
	float* __restrict__ m2x, float* __restrict__ m2y, float* __restrict__ m2z,
	float* __restrict__  Ms1_, float  Ms1_mul,
	float* __restrict__  Ms2_, float  Ms2_mul,
	int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float  ms1 = amul(Ms1_, Ms1_mul, i);
        float  ms2 = amul(Ms2_, Ms2_mul, i);
        float invMs = 1/(ms1+ms2);
        m0x[i] = (m1x[i]*ms1+m2x[i]*ms2)*invMs;
        m0y[i] = (m1y[i]*ms1+m2y[i]*ms2)*invMs;
        m0z[i] = (m1z[i]*ms1+m2z[i]*ms2)*invMs;
    }
}
