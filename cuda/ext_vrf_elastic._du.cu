#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "stencil.h"

extern "C"
__global__ void
calc_DU(float* __restrict__  dstx, float* __restrict__  dsty, float* __restrict__  dstz,
	float* __restrict__  sigmaxx, float* __restrict__  sigmayy, float* __restrict__  sigmaxy,
	float* __restrict__  ux, float* __restrict__  uy, float* __restrict__  uz,
	float* __restrict__  eta_, float eta_mul,
	float* __restrict__  rho_, float rho_mul,
	float* __restrict__ force_, float force_mul,
	float wx, float wy, float wz, int Nx, int Ny, int Nz)
	{

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;
	int I = idx(ix, iy, iz);
	if (ix >= Nx || iy >= Ny || iz >= Nz) {
		return;
	}
/*
	if (ix == 0) {  // x=0 is fixed
		dstx[I]=0.0f;
		dsty[I]=0.0f;
		dstz[I]=0.0f;
		return;
	}
*/
    float Eta = amul(eta_, eta_mul, I);
    float Rho = amul(rho_, rho_mul, I);
	float Force = amul(force_, force_mul, I);
	float dxsigmaxx,dysigmayy,dxsigmaxy,dysigmaxy;

	float Sigmaxx_0=sigmaxx[I];
	float Sigmayy_0=sigmayy[I];
	float Sigmaxy_0=sigmaxy[I];

	float Sigmaxx_v=0.0f;
	float Sigmaxy_v=0.0f;
	int J;
	if (ix+1<Nx) {
		J=idx(ix+1,iy,iz); // Stress 0 en nx+1 
		Sigmaxx_v=sigmaxx[J];
		Sigmaxy_v=sigmaxy[J];
	}
	dxsigmaxx=(Sigmaxx_v-Sigmaxx_0)*wx;
	dxsigmaxy=(Sigmaxy_v-Sigmaxy_0)*wx;
	Sigmaxy_v=0.0f;
	float Sigmayy_v=0.0f;
	if (iy+1<Ny) {
		J=idx(ix,iy+1,iz); // Stress 0 en ny+1
		Sigmayy_v=sigmayy[J];
		Sigmaxy_v=sigmaxy[J];
	}
	dysigmayy=(Sigmayy_v-Sigmayy_0)*wy;
	dysigmaxy=(Sigmaxy_v-Sigmaxy_0)*wy;

	dstx[I]=(dxsigmaxx+dysigmaxy-Eta*ux[I]+Force)/Rho;
	dsty[I]=(dysigmayy+dxsigmaxy-Eta*uy[I])/Rho;
	dstz[I]=0.0f;
	
	if (ix == 0 && iy==0 && ix==0) {  // x=0 is fixed
		dstx[I]=0.0f;
		dsty[I]=0.0f;
		dstz[I]=0.0f;
	}

	//if (ix==0) {dstx[I]=0.0f;dsty[I]=0.0f;dstz[I]=0.0f;}
}