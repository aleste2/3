#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "ext_LLB_functions.h"
#include "ext_AF_functions.h"

// Landau-Lifshitz torque for AF implementation PRB 100 054401 (2019)
// Uses me derived from eq. (9) of PRB 104413 (2012)
// Other equations from LowTempPhys 41 (9) 2015 due to better numerical stability

extern "C"

// New parameters
// mua,mub magnetic moment
// J0aa, J0bb, J0ab exchanges

__global__ void
LLBtorqueFerroUnified2(float* __restrict__  t1x, float* __restrict__  t1y, float* __restrict__  t1z,
	float* __restrict__  mx1, float* __restrict__  my1, float* __restrict__  mz1,
	float* __restrict__  hx1, float* __restrict__  hy1, float* __restrict__  hz1,
	float* __restrict__  alpha_, float alpha_mul,
	float* __restrict__  TCurie_, float TCurie_mul,
	float* __restrict__  Msat_, float Msat_mul,
	float* __restrict__  hth11x, float* __restrict__  hth11y, float* __restrict__  hth11z,
	float* __restrict__  hth21x, float* __restrict__  hth21y, float* __restrict__  hth21z,
	float* __restrict__  temp_, float temp_mul,
	float* __restrict__  te_,
	float* __restrict__  nv_, float nv_mul,
	float* __restrict__  mua_, float mua_mul,
	float* __restrict__  J0aa_, float J0aa_mul,
	float* __restrict__  deltaM_, float deltaM_mul,
	float* __restrict__ Qext_, float Qext_mul,
	int TTM,
	float* __restrict__ vol,
	int N) {

	const float kB=1.38064852e-23;
	//float mmin=0.0001; // Previously 0.01

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
        float3 m1 = {mx1[i], my1[i], mz1[i]};
        float3 H1 = {hx1[i], hy1[i], hz1[i]};
        float alpha = amul(alpha_, alpha_mul, i);
        float TCurie = amul(TCurie_, TCurie_mul, i);
        float Msat = amul(Msat_, Msat_mul, i);
        float nv = amul(nv_, nv_mul, i);
        float mua = amul(mua_, mua_mul, i);
        float J0aa = amul(J0aa_, J0aa_mul, i)*nv;
		float v = (vol == NULL? 1.0f: vol[i]);

		TCurie=J0aa/(3.0f*kB);

		float Qext = amul(Qext_, Qext_mul, i);
		float deltaM = amul(deltaM_, deltaM_mul, i);
		float temp;
        if (TTM==1) {temp = te_[i];} else{ temp=amul(temp_, temp_mul, i);}

        if (temp==0) temp=0.0001; // to avoid zero division...

        float3 hth1a = {hth11x[i], hth11y[i],hth11z[i]};
        float3 hth2a = {hth21x[i], hth21y[i],hth21z[i]};
        float3 torquea;
        float ma=sqrt(dot(m1,m1));
        // Parametros de LLB
		if ((v==0)||(ma==0))	{
			torquea = 0.0f*m1;
		} else {

			if (ma<0.01) ma=0.01;
			if (fabs(temp-TCurie)<0.1) {temp=TCurie+0.1;}  // To avoid errors arround T=Tc
			if (temp>1.5*TCurie) temp=1.5*TCurie; // To avoid numerical problems much above TC

			float alphaparA;
			float alphaperpA;
			alphaparA=2.0f*alpha*temp/(3.0f*TCurie);
			if (temp>TCurie) {
  				alphaperpA=alphaparA;
			} else {
				alphaperpA=alpha*(1.0f-temp/(3.0f*TCurie));
			}
			float gillba = 1.0f / (1.0f + alphaperpA * alphaperpA);
			float h_perp_scalea=sqrt((alphaperpA-alphaparA)/(alpha*alphaperpA*alphaperpA));
			float h_par_scalea=sqrt(alphaparA/alpha);

			float3 HMFA=H1+J0aa/mua*m1;
			float3 chi1=1/(kB*temp)*mua*HMFA;
			float mchi1=sqrt(dot(chi1,chi1));
			float L1=2.0f*alpha*gillba*kB*temp/mua;
			float G1p=L1*NL(mchi1)/(mchi1*DNL(mchi1));
			float G1pp=L1/2.0f*(mchi1/NL(mchi1)-1.0f);
			float3 m01=NL(mchi1)/mchi1*chi1;

			//torquea = -gillba*m1xH1+gillba*alphaparA/ma/ma*m1dotH1*m1-gillba*alphaperpA/ma/ma*(m1xm1xHtot1)+h_par_scalea*hth2a+mi;

			//h_par_scalea=0.0001;
			//if (i==1) printf("%e %e %e\n",HMFA.x,HMFA.y,HMFA.z);
			torquea = -gillba*cross(m1,HMFA)-G1p*(1-dot(m1,m01)/dot(m1,m1))*m1-G1pp/(dot(m1,m1))*(cross(m1,cross(m1,m01+h_perp_scalea*hth1a)))+h_par_scalea*hth2a;

		}

  		t1x[i] = torquea.x;
   		t1y[i] = torquea.y;
   		t1z[i] = torquea.z;
    }
}
