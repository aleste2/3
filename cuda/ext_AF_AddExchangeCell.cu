#include "amul.h"
#include "float3.h"
#include <stdint.h>

// normalize vector {vx, vy, vz} to unit length, unless length or vol are zero.
extern "C" __global__ void
addExchangeAFCell(float* __restrict__ dst1x, float* __restrict__ dst1y, float* __restrict__ dst1z,
	float* __restrict__ dst2x, float* __restrict__ dst2y, float* __restrict__ dst2z,
	float* __restrict__ m1x, float* __restrict__ m1y, float* __restrict__ m1z,
	float* __restrict__ m2x, float* __restrict__ m2y, float* __restrict__ m2z,
	float* __restrict__  Ms1_, float  Ms1_mul,
	float* __restrict__  Ms2_, float  Ms2_mul,
	float* __restrict__  Bex12_, float  Bex12_mul,
    	float* __restrict__  Bex21_, float  Bex21_mul,
	int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float invMs1 = inv_Msat(Ms1_, Ms1_mul, i);
        float invMs2 = inv_Msat(Ms2_, Ms2_mul, i);
        float bex12 = amul(Bex12_, Bex12_mul, i);
        float bex21 = amul(Bex21_, Bex21_mul, i);

        dst1x[i] += invMs1*bex12*m2x[i];
        dst1y[i] += invMs1*bex12*m2y[i];
        dst1z[i] += invMs1*bex12*m2z[i];
        dst2x[i] += invMs2*bex21*m1x[i];
        dst2y[i] += invMs2*bex21*m1y[i];
        dst2z[i] += invMs2*bex21*m1z[i];
    }
}
