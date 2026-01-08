#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"

extern "C"
__global__ void
evalM3TMtorque(float* __restrict__ dstx, float* __restrict__ dsty, float* __restrict__ dstz,
	float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
    float* __restrict__ musx, float* __restrict__ musy, float* __restrict__ musz,
    float* __restrict__ Te,
    float* __restrict__ Tc_, float Tc_mul,
    float* __restrict__ tausd_, float tausd_mul,
    int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    
    if (i < N) {
        float Tc = amul(Tc_, Tc_mul, i);
        float Tausd = amul(tausd_, tausd_mul, i);
        float Kb = 1.38064852e-23;
        float GammaLL=1.7595e11;

        if ((Tc>0.01)&&(mx[i]*mx[i]+my[i]*my[i]+mz[i]*mz[i]>0)) {
            //if (i==1) {printf("Entro mz:%e musz:%e Te:%e\n",mz[i],musz[i],Te[i]);}
           if (Tc<500) {printf("%e \n",Tc);}
            dstx[i]=0;//*(mx[i]-musx[i]/(2.0f*Kb*Tc))/Tausd*(1-mx[i]/tanh((2*mx[i]*Kb*Tc-musx[i])/(2*Kb*Te[i])));
            dsty[i]=0;//*(my[i]-musy[i]/(2.0f*Kb*Tc))/Tausd*(1-my[i]/tanh((2*my[i]*Kb*Tc-musy[i])/(2*Kb*Te[i])));
            dstz[i]=(mz[i]-musz[i]/(2.0f*Kb*Tc))/Tausd*(1-mz[i]/tanh((2*mz[i]*Kb*Tc-musz[i])/(2*Kb*Te[i])))/GammaLL;
        } else {
            dstx[i] = 0;
            dsty[i] = 0;
            dstz[i] = 0;
        }
    }
}
