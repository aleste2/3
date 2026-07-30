#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"
#include "exchange.h"
#include <math.h>

extern "C"

__global__ void
mcmove(float* __restrict__ mnewx, float* __restrict__ mnewy, float* __restrict__ mnewz,
    float* __restrict__ x, float* __restrict__ y, float* __restrict__ z,
    int iteration, 
    float delta,
    int Nx, int Ny, int Nz) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    int I = idx(ix, iy, iz);
    if ((mnewx[I]*mnewx[I]+mnewy[I]*mnewy[I]+mnewz[I]*mnewz[I])==0){
        return;
    }
    int change=((ix%2)^(iy%2))^(iz%2);
    if (change==iteration) {
        //if (I==0) printf("Rnd:%e %e %e %e %e %e\n",x[I],y[I],z[I],x[I+1],y[I+1],z[I+1]);
        /*
        float delta=0.005;
        float sum=fabs(x[I]+y[I]+z[I]-1.5);
        if (sum<0.005) delta=1.0;
        float mx=mnewx[I]+delta*(x[I]-0.5);
        float my=mnewy[I]+delta*(y[I]-0.5);
        float mz=mnewz[I]+delta*(z[I]-0.5);
        */
        //float delta=0.0025;
        float mx=mnewx[I]+delta*x[I];
        float my=mnewy[I]+delta*y[I];
        float mz=mnewz[I]+delta*z[I];
        float norm=sqrt(mx*mx+my*my+mz*mz);
        if (norm==0) {
            norm=1;
            mx=mnewx[I];
            my=mnewy[I];
            mz=mnewz[I];
        }

        mnewx[I]=mx/norm;
        mnewy[I]=my/norm;
        mnewz[I]=mz/norm;

        
/*
        float theta=acos(mnewz[I]);
        float phi=atan2(mnewy[I], mnewx[I]);
        float sigma = 0.5235f;
        mnewz[I]=cos(theta+angle1[I]*sigma);
        mnewx[I]=sin(theta+angle1[I]*sigma)*cos(phi+angle2[I]*sigma);
        mnewy[I]=sin(theta+angle1[I]*sigma)*sin(phi+angle2[I]*sigma);
*/
        }
    }