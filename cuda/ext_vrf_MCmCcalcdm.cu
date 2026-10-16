#include <stdint.h>
#include <math.h>
#include "amul.h"
#include "ext_LLB_functions.h"
#include "ext_AF_functions.h"

extern "C"

__global__ void
mCCalcdm(float* __restrict__ dst,
    float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
	float* __restrict__  Msat_, float Msat_mul,
    float* __restrict__  temp_, float temp_mul,
    float* __restrict__  TCurie_, float TCurie_mul,
    float* __restrict__  mua_, float mua_mul,
    float vol,
    int N) {


    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float  Ms  = amul(Msat_, Msat_mul, i);
        float  TCurie  = amul(TCurie_, TCurie_mul, i);
        float  temp  = amul(temp_, temp_mul, i);
        float  mua = amul(mua_, mua_mul, i);
        //const float mu0 = 1.256637e-6;
        const float kB=1.38064852e-23;
        //float m2=mx[i]*mx[i]+my[i]*my[i]+mz[i]*mz[i];
        //float me,xpar;
        float sigma;
        if (temp<TCurie) {
            /*
            me=pow(1.0f-pow(temp/TCurie,1.23318f),0.433936f);
            if (me<0.001f) me=0.001f;
            float chiA=3.0f*TCurie/temp*me;
            xpar=(mua/kB)*DNL(chiA)/((temp-3.0f*TCurie*DNL(chiA)));
            //dst[i]+=Ms*pow((m2-me*me),2)/(8.0f*xpar*me*me);
            */
            //float me=meLLB(temp, TCurie);  // From LLB functions
            //float xpar=chiParLLB(temp, me, mua);
            float me=meMF(temp, TCurie);  // from LLB functions
            float xpar=chiParMF(temp,TCurie, me, mua);
            sigma=2.0f*me*sqrt(xpar*kB*temp/(vol*Ms));
            if (sigma>0.03f) {sigma=0.03f;}
        } else {
            sigma=0.03f;
        }
        dst[i]=sigma*2.0f*(dst[i]-0.5f);
    }
}