#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "stencil.h"

extern "C"
__global__ void
addStrain(float* __restrict__  dstx, float* __restrict__  dsty, float* __restrict__  dstz,
    float* __restrict__  ux, float* __restrict__  uy, float* __restrict__  uz,
    float wx, float wy, float wz, int Nx, int Ny, int Nz)
    {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    int I = idx(ix, iy, iz);

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
		return;
	}

    float exx,exy,eyy;

    if (ix==0) {
        exx=(ux[idx(ix+1,iy,iz)]-ux[I])*wx;
        exy=0.5f*(uy[idx(ix+1,iy,iz)]-uy[I])*wx;
    } else if (ix==Nx-1) {
        exx=(ux[I]-ux[idx(ix-1,iy,iz)])*wx;
        exy=0.5f*(uy[I]-uy[idx(ix-1,iy,iz)])*wx;
    } else {
        exx=0.5*(ux[idx(ix+1,iy,iz)]-ux[idx(ix-1,iy,iz)])*wx;
        exy=0.5*0.5f*(uy[idx(ix+1,iy,iz)]-uy[idx(ix-1,iy,iz)])*wx;
    }

    if (iy==0) {
        eyy=(uy[idx(ix,iy+1,iz)]-uy[I])*wy;
        exy+=0.5f*(ux[idx(ix,iy+1,iz)]-ux[I])*wy;
    } else if (iy==Ny-1) {
        eyy=(uy[I]-uy[idx(ix,iy-1,iz)])*wy;
        exy+=0.5f*(ux[I]-ux[idx(ix,iy-1,iz)])*wy;
    } else {
        eyy=0.5*(uy[idx(ix,iy+1,iz)]-uy[idx(ix,iy-1,iz)])*wy;
        exy+=0.5*0.5f*(ux[idx(ix,iy+1,iz)]-ux[idx(ix,iy-1,iz)])*wy;
    }

    dstx[I]+=exx;
    dsty[I]+=eyy;
    dstz[I]+=exy;
    //if (I==200) printf("Entro %e \n",dstx[I]);
    }