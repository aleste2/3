#include "amul.h"
#include "float3.h"
#include <stdint.h>

// Landau-Lifshitz torque.
extern "C" 

__global__ void
initmLLB(float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
        float* __restrict__  temp_, float temp_mul,
        float* __restrict__  TCurie_, float TCurie_mul,
        int N,int Langevin) {
 
    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {

        float3 m = {mx[i], my[i], mz[i]};
        float TCurie = amul(TCurie_, TCurie_mul, i);
        float temp = amul(temp_, temp_mul, i);
        if (temp==0) temp=0.0001; // to avoid zero division...
        float m2=dot(m,m);
 
        if ((m2!=0)&&(TCurie!=0))
        {
	 if (temp<=TCurie)  // T<Tc
         {
         	float me;
		if (!Langevin) {
				me=pow(1.0f-pow(temp/TCurie,3.49f),0.54f);
				} else {
				//me=pow(1.0f-pow(temp/TCurie,1.23329f),0.43392f);
                me=pow(1.0f-pow(temp/TCurie,1.23318f),0.433936f);
				}
		mx[i]=mx[i]*(me/pow(m2,0.5f));
		my[i]=my[i]*(me/pow(m2,0.5f));
		mz[i]=mz[i]*(me/pow(m2,0.5f));
         }
         else        //T>Tc
         {
         	float me=0.0001;
		mx[i]=mx[i]*(me/pow(m2,0.5f));
		my[i]=my[i]*(me/pow(m2,0.5f));
		mz[i]=mz[i]*(me/pow(m2,0.5f));
         };
	}
    }
}
