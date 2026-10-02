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
        float me,xpar;
        if (temp<TCurie) {
            me=pow(1.0f-pow(temp/TCurie,1.23318f),0.433936f);
            if (me<0.001) me=0.001;
            float chiA=3.0f*TCurie/temp*me;
            xpar=(mua/kB)*DNL(chiA)/((temp-3.0f*TCurie*DNL(chiA)));
            //dst[i]+=Ms*pow((m2-me*me),2)/(8.0f*xpar*me*me);
        } else {
            me=0.001;
            float chiA=3.0f*TCurie/temp*me;
            xpar=(mua/kB)*DNL(chiA)/((temp-3.0f*TCurie*DNL(chiA)));
            //dst[i]+=mu0*Ms*Ms/(2.0f*xpar)*(3.0f*temp/(10.0f*TCurie)*m2*m2+(temp/TCurie-1.0f)*m2);
        }
        float sigma=2*me*sqrt(xpar*kB*temp/(vol*Ms));
        if (sigma>0.03) {sigma=0.03;}
        dst[i]=1.0f+sigma*2.0*(dst[i]-0.5f);
    }
}