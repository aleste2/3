#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"
#include "exchange.h"
#include <math.h>

extern "C"

__global__ void
mccompare(
    float* __restrict__ uold,float* __restrict__ unew,
    float* __restrict__ moldx, float* __restrict__ moldy, float* __restrict__ moldz,
    float* __restrict__ mnewx, float* __restrict__ mnewy, float* __restrict__ mnewz,
    float* __restrict__ rnd,
    float* __restrict__  temp_, float  temp_mul,
    float* __restrict__ changed,
    int iteration, 
    int Nx, int Ny, int Nz,
    float wx,float  wy, float wz,
    float cells) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    //float vol=cells*wx*wy*wz;
    float vol=wx*wy*wz;
    
    int I = idx(ix, iy, iz);
    float mmo=sqrt(moldx[I]*moldx[I]+moldy[I]*moldy[I]+moldz[I]*moldz[I]);
    if (mmo==0) {
        return;
    }

    int change=((ix%2)^(iy%2))^(iz%2);
    if (change==iteration) {
        changed[I]=1;
        float temp = amul(temp_, temp_mul, I);
/*
        if (I==0) {
            printf("DeltaU: %e: KbT: %e Rnd: %e:\n",(unew[I]-uold[I])*wx*wy*wz,temp*1.38e-23,rnd[I]);
        }
*/

/*
        if (unew[I]>=uold[I]) {
            float R;
            if (temp==0) {
                R=0.0f;
            } else {
                float mmn=sqrt(mnewx[I]*mnewx[I]+mnewy[I]*mnewy[I]+mnewz[I]*mnewz[I]);
                                float prefactor=pow(mmn/mmo,4); // For LLB
                //prefactor=1.0f;

                R=prefactor*exp(-(unew[I]-uold[I])*vol/(temp*1.38e-23));
                //if (I==0) printf("%e %e\n",R,(unew[I]-uold[I])*wx*wy*wz/(temp*1.38e-23));
            }
            if (rnd[I]>R) {
                mnewx[I]=moldx[I];
                mnewy[I]=moldy[I];
                mnewz[I]=moldz[I];
                changed[I]=0;
            }
        }
*/

// Claude code

        float R;
        if (temp > 0) {
            float mmn = sqrtf(mnewx[I]*mnewx[I]+mnewy[I]*mnewy[I]+mnewz[I]*mnewz[I]);
            R = powf(mmn/mmo, 2.0f) * expf(-(unew[I]-uold[I])*vol/(temp*1.38064852e-23f)); //pow 2 y no 4 segun claude
        } else {
            R = (unew[I] <= uold[I]) ? 1.0f : 0.0f;
        }
        if (!(rnd[I] <= R)) {
            mnewx[I]=moldx[I];
            mnewy[I]=moldy[I];
            mnewz[I]=moldz[I];
            changed[I]=0;
        }


    }
}