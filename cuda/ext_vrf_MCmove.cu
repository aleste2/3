#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"
#include "exchange.h"
#include <math.h>

extern "C"

__global__ void
mcmove(float* __restrict__ mnewx, float* __restrict__ mnewy, float* __restrict__ mnewz,
    float* __restrict__ x, float* __restrict__ y, float* __restrict__ z,float* __restrict__ dmm,
    int iteration, 
    float delta,
    float norm0,
    int Nx, int Ny, int Nz) {

        
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    int I = idx(ix, iy, iz);
    float oldm=sqrt(mnewx[I]*mnewx[I]+mnewy[I]*mnewy[I]+mnewz[I]*mnewz[I]);
    if (oldm==0){
        return;
    }
    int change=((ix%2)^(iy%2))^(iz%2);
    if (change==iteration) {
        //delta=0;
        float mx=mnewx[I]/oldm+delta*x[I];
        float my=mnewy[I]/oldm+delta*y[I];
        float mz=mnewz[I]/oldm+delta*z[I];
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

        // LLB
        if (norm0==0){
            //float oldm=sqrt(mnewx[I]*mnewx[I]+mnewy[I]*mnewy[I]+mnewz[I]*mnewz[I]);
            //float newm=(oldm+oldm*delta*(x[I]+y[I]+z[I]));
            //float newm=oldm+0.01*(x[I]+y[I]+z[I]);
            float newm=oldm+dmm[I];
            if (newm>1) newm=1.0f;
            if (newm<=0.0001) newm=0.0001;  
            mnewx[I]=mnewx[I]*newm;
            mnewy[I]=mnewy[I]*newm;
            mnewz[I]=mnewz[I]*newm;
 
        }

 //if (I==0) printf("%e %e %e %e %e %e %e\n",mnewx[I],mnewy[I],mnewz[I],mx,my,mz,norm);

        //mnewx[I]=mx/norm;
        //mnewy[I]=my/norm;
        //mnewz[I]=mz/norm;

        }

 /*
 new code




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

        //float delta=0.0025;
        float mx=mnewx[I]+delta*x[I];
        float my=mnewy[I]+delta*y[I];
        float mz=mnewz[I]+delta*z[I];
        float norm=sqrt(mx*mx+my*my+mz*mz);

        // LLG
        if (norm0==1) {
            if (norm==0) norm=1;
            mx=mnewx[I];
            my=mnewy[I];
            mz=mnewz[I];
        }
        // LLB
        //if (I==0) printf("m:%e delat:%e\n",norm, delta*(x[I]+y[I]+z[I])/norm);
        if (norm0==0) norm=1;
        
        mnewx[I]=mx/norm;
        mnewy[I]=my/norm;
        mnewz[I]=mz/norm;
        
        }

        */
    }