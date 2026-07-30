#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"
#include "exchange.h"

extern "C"


 __global__ void
ttmplusU(
    float* __restrict__ u0,
    float* __restrict__ u1,
    float* __restrict__  Te_,
    float* __restrict__ Tc_, float Tc_mul,
    float dt,
    float* __restrict__ Ce_, float Ce_mul,
    int N
) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
        float Tc = amul(Tc_, Tc_mul, i);
        float Te=Te_[i];
        float Ce = amul(Ce_, Ce_mul, i);
        if (Te!=0) {Ce=Ce*Te/300.0;} else {Ce=Ce*1.0/300.0;}  // To account for temperature dependence. Cl is almost constat at T>TD

        Te_[i]+=(u0[i]-u1[i])/Ce;

        //if (i==0) printf("%e %e\n",u0[i]-u1[i],(u0[i]-u1[i])/Ce);
    }
}
