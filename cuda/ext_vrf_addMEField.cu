#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "stencil.h"

extern "C"
__global__ void
addMEField(float* __restrict__  dstx, float* __restrict__  dsty, float* __restrict__  dstz,
    float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
    float* __restrict__  ux, float* __restrict__  uy, float* __restrict__  uz,
    float* __restrict__  c11_, float c11_mul,
    float* __restrict__  c12_, float c12_mul,
    float* __restrict__ c44_, float c44_mul,
    float* __restrict__ b1_, float b1_mul,
    float* __restrict__ b2_, float b2_mul,
    float* __restrict__ ms_, float ms_mul,
    float wx, float wy, float wz, int Nx, int Ny, int Nz)
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

    //float lambda100=-2.0f/3.0f*B1/(C11-C12);
    //float lambda111=-1.0f/3.0f*B2/C44;

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
/*
    dstx[I]+=3.0f/Ms*(lambda100*exx*mx[I]+lambda111*(eyy)*my[I]);
    dsty[I]+=3.0f/Ms*(lambda100*exy*mx[I]+lambda111*(eyy)*my[I]);
    dstz[I]+=0.0f;
*/
    dstx[I]+=-2/Ms*(B1*exx*mx[I]+B2*exy*my[I]);
    dsty[I]+=-2/Ms*(B1*eyy*my[I]+B2*exy*mx[I]);
    dstz[I]+=0.0f;
    //if (I==200) printf("Entro %e \n",dstx[I]);
    }