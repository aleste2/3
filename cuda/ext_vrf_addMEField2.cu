#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "stencil.h"

extern "C"
__global__ void
addMEField2(float* __restrict__  dstx, float* __restrict__  dsty, float* __restrict__  dstz,
    float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
    float* __restrict__  sigmaxx, float* __restrict__  sigmayy, float* __restrict__  sigmaxy,
    float* __restrict__  c11_, float c11_mul,
    float* __restrict__  c12_, float c12_mul,
    float* __restrict__ c44_, float c44_mul,
    float* __restrict__ b1_, float b1_mul,
    float* __restrict__ b2_, float b2_mul,
    float* __restrict__ ms_, float ms_mul,
    float vol, int Nx, int Ny, int Nz)
    {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    int I = idx(ix, iy, iz);

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
		return;
	}

    float C11 = amul(c11_, c11_mul, I);
    float C12 = amul(c12_, c12_mul, I);
    float C44 = amul(c44_, c44_mul, I);
    float B1 = amul(b1_, b1_mul, I);
    float B2 = amul(b2_, b2_mul, I);
    float Ms = amul(ms_, ms_mul, I);

    float lambda100=-2.0f/3.0f*B1/(C11-C12);
    float lambda111=-1.0f/3.0f*B2/C44;

    dstx[I]+=-3.0f/Ms*(lambda100*sigmaxx[I]*mx[I]+lambda111*sigmaxy[I]*my[I]);
    dsty[I]+=-3.0f/Ms*(lambda100*sigmaxy[I]*mx[I]+lambda111*sigmayy[I]*my[I]);
    dstz[I]+=0.0f;
    //if (I==200) printf("Entro %e \n",dstx[I]);
    }