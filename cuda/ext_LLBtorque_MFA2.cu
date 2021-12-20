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
LLBtorqueMFA2(float* __restrict__  t1x, float* __restrict__  t1y, float* __restrict__  t1z,
	float* __restrict__  mx1, float* __restrict__  my1, float* __restrict__  mz1,
	float* __restrict__  hx1, float* __restrict__  hy1, float* __restrict__  hz1,
	float* __restrict__  alpha_, float alpha_mul,
	float* __restrict__  Msat_, float Msat_mul,
	float* __restrict__  hth11x, float* __restrict__  hth11y, float* __restrict__  hth11z,
	float* __restrict__  hth21x, float* __restrict__  hth21y, float* __restrict__  hth21z,
	float* __restrict__  temp_, float temp_mul,
	float* __restrict__  nv_, float nv_mul,
	float* __restrict__  mua_, float mua_mul,
	float* __restrict__  J0aa_, float J0aa_mul,
	int N) {

    const float kB=1.38064852e-23;

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
        float3 m1 = {mx1[i], my1[i], mz1[i]};
        float3 H1 = {hx1[i], hy1[i], hz1[i]};
        float alphaa = amul(alpha_, alpha_mul, i);
        float Msat = amul(Msat_, Msat_mul, i);
        float nv = amul(nv_, nv_mul, i);
        float mua = amul(mua_, mua_mul, i);
        float J0aa = amul(J0aa_, J0aa_mul, i)*nv;
        float temp = amul(temp_, temp_mul, i);

        if (temp==0) temp=0.0001; // to avoid zero division...

        float3 hth1a = {hth11x[i], hth11y[i],hth11z[i]};
        float3 hth2a = {hth21x[i], hth21y[i],hth21z[i]};
        float3 torquea;

        // Parametros de LLB

        float ma=sqrt(dot(m1,m1));
        if (ma==0)	{
					torquea = 0.0f*m1;
  			} else {

					float J01 = J0aa;
					float TCurie = (J01)/(3.0f*kB); // Eq (9) LowTempPhys 41 (9) 2015
					if ((fabs(temp-TCurie)<0.007*TCurie)&&(temp<TCurie)) {temp=0.993f*TCurie;}  // To avoid errors arround T=Tc
					if (temp>2.0*TCurie) temp=2.0*TCurie; // To avoid numerical problems much above TC

					float alphaparA;
					float alphaperpA;
					if (temp<TCurie) {
						alphaparA=2.0f*alphaa*temp/(3.0f*TCurie);
						alphaperpA=alphaa*(1.0f-temp/(3*TCurie));
					} else {
						alphaparA=2.0f*alphaa*temp/(3.0f*TCurie);
						alphaperpA=alphaparA;
					}

					float3 HMFA1=J0aa/mua*m1;

					float beta=1.0f/(kB*temp);
					float3 chiA=beta*mua*HMFA1;
					float mchiA=sqrt(dot(chiA,chiA));
					float3 m0A;
					float lambdaA;
					float GammaApara;
					float GammaAperp;
					if (temp<TCurie/5){   // For Low Temp do not restict Langevin/Dlangevin small values
						m0A=dL0(mchiA)/mchiA*chiA;
						lambdaA=2.0f*alphaa*kB*temp/mua;
						GammaApara=lambdaA*dL0(mchiA)/(mchiA*dLder0(mchiA));
						GammaAperp=lambdaA/2.0f*(mchiA/dL0(mchiA)-1.0f);
					} else {
						m0A=dL(mchiA)/mchiA*chiA;
						lambdaA=2.0f*alphaa*kB*temp/mua;
						GammaApara=lambdaA*dL(mchiA)/(mchiA*dLder(mchiA));
						GammaAperp=lambdaA/2.0f*(mchiA/dL(mchiA)-1.0f);
					}

		float h_perp_scalea=sqrt((alphaperpA-alphaparA)/(alphaa*alphaperpA*alphaperpA));
		float h_par_scalea=sqrt(alphaparA/alphaa);

		// Recuperamos todas las contribuciones
		H1=H1+J0aa/mua*m1;

		float3 htotA=m0A+h_perp_scalea*hth1a;
    float3 m1xHmfa = cross(m1, H1);
    float m1dotm0a = dot(m1, m0A);
    float3 m1xHtot1 = cross(m1, htotA);
    float3 m1xm1xHtot1 = cross(m1, m1xHtot1);
    float gillba = 1.0f / (1.0f + alphaa * alphaa);

		torquea = -gillba*(m1xHmfa+GammaApara*(1.0f-m1dotm0a/ma/ma)*m1+GammaAperp/ma/ma*(m1xm1xHtot1))+h_par_scalea*hth2a;
    }
    t1x[i] = torquea.x;
    t1y[i] = torquea.y;
    t1z[i] = torquea.z;
    }
}
