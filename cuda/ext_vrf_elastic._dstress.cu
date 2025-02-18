#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "stencil.h"

extern "C"
__global__ void
calc_Sigma(float* __restrict__  dstx, float* __restrict__  dsty, float* __restrict__  dstz,
	float* __restrict__  ux, float* __restrict__  uy, float* __restrict__  uz,
	float* __restrict__  c11_, float c11_mul,
	float* __restrict__  c12_, float c12_mul,
	float* __restrict__ c44_, float c44_mul,
	float wx, float wy, float wz, int Nx, int Ny, int Nz)
	{

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;
	int I = idx(ix, iy, iz);
	if (ix == 0 || iy == 0|| ix >= Nx || iy >= Ny || iz >= Nz) { // add iz==0&&Nz>1 in the future
		return;
	}

    float C11 = amul(c11_, c11_mul, I);
    float C12 = amul(c12_, c12_mul, I);
    float C44 = amul(c44_, c44_mul, I);

    float vx0=ux[I];
    float vy0=uy[I];

	float vx_v=0.0f;
	float vy_v=0.0f;

    float dxvx,dxvy,dyvx,dyvy;

	int J;
	J=idx(ix-1,iy,iz); // Stress 0 en nx+1 
	vx_v=ux[J];
    vy_v=uy[J];
    dxvx=(vx0-vx_v)*wx;
    dxvy=(vy0-vy_v)*wx;
    
    J=idx(ix,iy-1,iz); // Stress 0 en ny+1
	vx_v=ux[J];
    vy_v=uy[J];
    dyvx=(vx0-vx_v)*wy;
    dyvy=(vy0-vy_v)*wy;

	dstx[I]=C11*dxvx+C12*dyvy;
	dsty[I]=C11*dyvy+C12*dxvx;
	dstz[I]=C44*(dyvx+dxvy);

}