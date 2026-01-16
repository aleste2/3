#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"
#include "exchange.h"

extern "C"


 __global__ void
ttmplus(
    float* __restrict__ dy11x, float* __restrict__ dy11y, float* __restrict__ dy11z,
    float* __restrict__ dy1x, float* __restrict__ dy1y, float* __restrict__ dy1z,
    float* __restrict__  Te_,
    float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
    float* __restrict__ Tc_, float Tc_mul,
    float* __restrict__ rho_, float rho_mul,
    float dt,
    float* __restrict__ Ce_, float Ce_mul,
    int N
) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
     float3 m0={mx[i], my[i], mz[i]};
     float mm=dot(m0,m0);
     if (mm!=0) {

        float GammaLL= 1.7595e11;
        float moldx=mx[i]-(dt*GammaLL)*(dy1x[i]); // LL constant!
        float moldy=my[i]-(dt*GammaLL)*(dy1y[i]); // LL constant!
        float moldz=mz[i]-(dt*GammaLL)*(dy1z[i]); // LL constant!
        float mnewx=mx[i]+(dt*GammaLL)*(dy11x[i]*0.5f-0.5f*dy1x[i]); // LL constant!
        float mnewy=my[i]+(dt*GammaLL)*(dy11y[i]*0.5f-0.5f*dy1y[i]); // LL constant!
        float mnewz=mz[i]+(dt*GammaLL)*(dy11z[i]*0.5f-0.5f*dy1z[i]); // LL constant!
        float mmnew=sqrt(mnewx*mnewx+mnewy*mnewy+mnewz*mnewz);
        float mmold=sqrt(moldx*moldx+moldy*moldy+moldz*moldz);
        float dmdt=0*(mmnew-mmold)/dt;


        float kb=1.380649e-23;
        float Tc = amul(Tc_, Tc_mul, i);
        float rho = amul(rho_, rho_mul, i);

        float Te=Te_[i];
        float Ce = amul(Ce_, Ce_mul, i);
        if (Te!=0) {Ce=Ce*Te/300.0;} else {Ce=Ce*1.0/300.0;}  // To account for temperature dependence. Cl is almost constat at T>TD

        Te_[i]+=(2.0f*kb*Tc*rho*mmold*dmdt*dt)/Ce;

     }
    }
}
