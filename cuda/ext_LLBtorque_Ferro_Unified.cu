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
LLBtorqueFerroUnified(float* __restrict__  t1x, float* __restrict__  t1y, float* __restrict__  t1z,
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
	int N) {

		const float kB=1.38064852e-23;
		float mmin=0.0001; // Previously 0.01

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

				float Qext = amul(Qext_, Qext_mul, i);
				float deltaM = amul(deltaM_, deltaM_mul, i);
				float temp;
        if (TTM==1) {temp = te_[i];} else{ temp=amul(temp_, temp_mul, i);}

        if (temp==0) temp=0.0001; // to avoid zero division...

        float3 hth1a = {hth11x[i], hth11y[i],hth11z[i]};
        float3 hth2a = {hth21x[i], hth21y[i],hth21z[i]};
        float3 torquea;

        // Parametros de LLB
        float ma=sqrt(dot(m1,m1));
				if (ma<0.01) ma=0.01;
        if (ma==0)	{
					torquea = 0.0f*m1;
 				} else {

					if ((fabs(temp-TCurie)<0.001*TCurie)&&(temp<TCurie)) {temp=0.999f*TCurie;}  // To avoid errors arround T=Tc
					if (temp>1.5*TCurie) temp=1.5*TCurie; // To avoid numerical problems much above TC

					float mea;
					if (temp<TCurie) {
						mea=pow(1.0f-pow(temp/TCurie,1.23318f),0.433936f);
					} else	{
						mea=mmin;
					}
					if (mea<mmin) {mea=mmin;}

					float chiA=TCurie/temp*mea;  // Arguments of Langevin function
					float xpara;
					if (temp<=TCurie) {
						xpara=mua/(kB*temp)*DNL(chiA)/((1.0f-TCurie/temp*DNL(chiA)));
					} else { // T>Tc DNL=1/3
						xpara=mua/(kB*temp)/3.0f/((1.0f-TCurie/temp/3.0f));
					}

					//if (xpara<=0.0) xpara=0.0001f;

					float lambdaA=(1.0f/xpara);
    			float3 heffa;
					float alphaparA;
					float alphaperpA;
					alphaparA=2.0f*alpha*temp/(3.0f*TCurie);
    			if (temp>=TCurie) {
		  			heffa = -lambdaA*m1;
						alphaperpA=alphaparA;
    			} else {
      			heffa = -lambdaA*((ma-mea)/mea)*m1;
						alphaperpA=alpha*(1.0f-temp/(3.0f*TCurie));
    			}

					float h_perp_scalea=sqrt((alphaperpA-alphaparA)/(alpha*alphaperpA*alphaperpA));
					float h_par_scalea=sqrt(alphaparA/alpha);

					H1=H1+heffa;

    			float3 htot1=H1+h_perp_scalea*hth1a;

    			float3 m1xH1 = cross(m1, H1);
    			float m1dotH1 = dot(m1, H1);
    			float3 m1xHtot1 = cross(m1, htot1);
    			float3 m1xm1xHtot1 = cross(m1, m1xHtot1);
    			float gillba = 1.0f / (1.0f + alphaperpA * alphaperpA);

					// Direct laser moment induction
					float3 mi = {0.0f,0.0f,(Qext/1e20)*deltaM};

					torquea = -gillba*m1xH1+gillba*alphaparA/ma/ma*m1dotH1*m1-gillba*alphaperpA/ma/ma*(m1xm1xHtot1)+h_par_scalea*hth2a+mi;
				}

    		t1x[i] = torquea.x;
    		t1y[i] = torquea.y;
    		t1z[i] = torquea.z;
    }
}
