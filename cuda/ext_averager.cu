#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "math.h"

// See exchange.go for more details.
extern "C" __global__ void
averager(float* __restrict__ dstx, float* __restrict__ dsty, float* __restrict__ dstz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float* __restrict__ mask,
            int Nv,
            int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    int I0 = idx(ix, iy, iz);
    dstx[I0]=0;
    dsty[I0]=0;
    dstz[I0]=0;

    // No magnetic cell, no average
    if (mask[I0]==0) {
        return;
     }

    float ncell=0;
    for (int ii=-Nv;ii<=Nv;ii++) {
        for (int jj=-Nv;jj<=Nv;jj++) {
            for (int kk=-Nv;kk<=Nv;kk++) {
                int iix=ix+ii;
                int iiy=iy+jj;
                int iiz=iz+kk;
                if ((iix>=0)&&(iix<Nx)&&(iiy>=0)&&(iiy<Ny)&&(iiz>=0)&&(iiz<Nz)) {
                    int I = idx(iix, iiy, iiz);
                    if (mask[I]!=0){
                    dstx[I0]+=mx[I];
                    dsty[I0]+=my[I];
                    dstz[I0]+=mz[I];
                        ncell++;
                    }
                }
            }
        }
    }
    if (ncell>0) {
        dstx[I0]=dstx[I0]/ncell;
        dsty[I0]=dsty[I0]/ncell;
        dstz[I0]=dstz[I0]/ncell;
    }
}

