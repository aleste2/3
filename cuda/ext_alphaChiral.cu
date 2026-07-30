#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// AlphaChiralCalculation
extern "C" __global__ void
alphachiralLocal(float* __restrict__ dst,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float* __restrict__ alphaC_, float alphaC_mul,
            float* __restrict__ Ku1_, float Ku1_mul,
            float* __restrict__ aLUT2d, uint8_t* __restrict__ regions,
            float wx, float wy, float wz, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // central cell
    int I = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[I], my[I], mz[I]);

    if (is0(m0)) {
        return;
    }

    float  alphaC  = amul(alphaC_, alphaC_mul, I);
    float  Ku1  = amul(Ku1_, Ku1_mul, I);
    uint8_t r0 = regions[I];
    float  Aex = aLUT2d[symidx(r0, r0)];
    float fx=1.0;
    float fy=1.0;

    int i_;    // neighbor index
    float3 mXm,mYm,mXp,mYp; // neighbours XY

    // left neighbor
    i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
    mXm  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    if (is0(mXm)) { mXm=m0;fx=2.0;}
    // right neighbor
    i_  = idx(hclampx(ix+1), iy, iz);
    mXp  = make_float3(mx[i_], my[i_], mz[i_]);
    if (is0(mXp)) { mXp=m0;fx=2.0;}

    // back neighbor
    i_  = idx(ix, lclampy(iy-1), iz);
    mYm  = make_float3(mx[i_], my[i_], mz[i_]);
    if (is0(mYm)) { mYm=m0;fy=2.0;}

   // front neighbor
    i_  = idx(ix, hclampy(iy+1), iz);
    mYp  = make_float3(mx[i_], my[i_], mz[i_]);
    if (is0(mYp)) { mYp=m0;fy=2.0;}

    dst[I]=alphaC*sqrt(Aex/Ku1)*(m0.x*fx*(mXp.z-mXm.z)/(2.0*wx)+m0.y*fy*(mYp.z-mYm.z)/(2.0*wy)-m0.z*(fx*(mXp.x-mXm.x)/(2.0*wx)+fy*(mYp.y-mYm.y)/(2.0*wy)));

}
