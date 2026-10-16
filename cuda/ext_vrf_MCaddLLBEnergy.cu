#include <stdint.h>
#include <math.h>
#include "amul.h"
#include "ext_LLB_functions.h"
#include "ext_AF_functions.h"

extern "C"

__global__ void
addLLBEnergyDensity(float* __restrict__ dst,
    float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
	float* __restrict__  Msat_, float Msat_mul,
    float* __restrict__  temp_, float temp_mul,
    float* __restrict__  TCurie_, float TCurie_mul,
    float* __restrict__  mua_, float mua_mul,
    int N) {

    //printf("Entro\n");

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float  Ms  = amul(Msat_, Msat_mul, i);
        float  TCurie  = amul(TCurie_, TCurie_mul, i);
        float  temp  = amul(temp_, temp_mul, i);
        float  mua = amul(mua_, mua_mul, i);
        //const float mu0 = 1.256637e-6;
        const float kB=1.38064852e-23;
        float m2=mx[i]*mx[i]+my[i]*my[i]+mz[i]*mz[i];
      
        if (temp<TCurie) {
            /*
            float me=pow(1.0f-pow(temp/TCurie,1.23318f),0.433936f);
            if (me<0.001) me=0.001;
            float chiA=3.0f*TCurie/temp*me;
            float xpar=(mua/kB)*DNL(chiA)/((temp-3.0f*TCurie*DNL(chiA)));
            //if (xpar>0.0005) xpar=0.0005;
            //if (xpar>0.001) xpar=0.001;

            //(if (i==1) printf("%e %e %e %e Ms:%e Tc:%e T:%e\n",dst[i],mu0*Ms*Ms*pow((m2-me*me),2)/(8.0f*xpar),xpar,me,Ms,TCurie,temp);

            //dst[i]+=mu0*Ms*Ms*pow((m2-me*me),2)/(8.0f*xpar);
            //dst[i]+=mu0*Ms*Ms*pow((m2-me*me),2)/(2.0f*xpar);
            dst[i]+=Ms*pow((m2-me*me),2)/(8.0f*xpar*me*me);
            //dst[i]+=Ms*pow((m2-me*me),2)/(2.0f*xpar*me*me);
            */
           
           //float me=meLLB(temp, TCurie);  // from LLB functions
           //float xpar=chiParLLB(temp, me, mua);
           float me=meMF(temp, TCurie);  // from LLB functions
           float xpar=chiParMF(temp,TCurie, me, mua);
           dst[i]+=Ms*pow((m2-me*me),2)/(8.0f*xpar*me*me);

        } else {
            //float chiA=3.0f*TCurie/temp*0.001;
            //float xpar=(mua/kB)*DNL(chiA)/((temp-3.0f*TCurie*DNL(chiA)));
            //float xpar=(mua/kB)*0.33333333333f/(temp-TCurie);
            //if (xpar>0.001) xpar=0.001;
            //dst[i]+=Ms/(2.0f*xpar)*(3.0f*temp/(10.0f*TCurie)*m2*m2+(temp/TCurie-1.0f)*m2);
            //dst[i]+=Ms/(2.0f*xpar)*(m2+m2*m2*3.0f*TCurie/(10.0*(temp-TCurie)));
            dst[i] += 3.0f*kB*Ms/(2.0f*mua) * ((temp-TCurie)*m2 + 0.3f*TCurie*m2*m2);// Claude
        }
    }
}