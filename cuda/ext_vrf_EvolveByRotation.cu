#include "amul.h"
#include "float3.h"
#include <stdint.h>

// m evolution by rotation
extern "C" __global__ void

evolveAtomisticByRotation(
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__  hx, float* __restrict__  hy, float* __restrict__  hz,
          float* __restrict__  alpha_, float alpha_mul,
          float* __restrict__  lande_, float lande_mul,
          float dt,
          int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 m = {mx[i], my[i], mz[i]};
        float mm=dot(m,m);
        if (mm==0) {return;}
        float3 H = {hx[i], hy[i], hz[i]};
        float alpha = amul(alpha_, alpha_mul, i);
        float lande = amul(lande_, lande_mul, i);
        if (lande==0) lande=2.0;

        float3 Omega=(lande/2.0*dt/(1+alpha*alpha))*(H+alpha*cross(m,H));
        float mOmega=sqrt(dot(Omega,Omega));

        //if (i==1) printf("%e\n",mOmega);

        float cosO=cos(mOmega);
        float sinO=sin(mOmega);
        float3 nOmega=(1.0f/mOmega)*Omega;

        //if (i==1) printf("%e %e %e\n",nOmega.x,nOmega.y,nOmega.z);

        float3 p1=cosO*m;
        float3 p2=sinO*cross(nOmega,m);
        float3 p3=(dot(nOmega,m)*(1.0f-cosO))*nOmega;
        float3 mnew=p1+p2+p3;

        //float3 mnew=cosO*m+(sinO*cross(nOmega,m))+(dot(nOmega,m)*(1.0f-cosO))*nOmega;

        mx[i]=mnew.x;
        my[i]=mnew.y;
        mz[i]=mnew.z;
    }
}

