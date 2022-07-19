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
LLBtorqueAFMFA3(float* __restrict__  t1x, float* __restrict__  t1y, float* __restrict__  t1z,
	float* __restrict__  mx1, float* __restrict__  my1, float* __restrict__  mz1,
	float* __restrict__  t2x, float* __restrict__  t2y, float* __restrict__  t2z,
	float* __restrict__  mx2, float* __restrict__  my2, float* __restrict__  mz2,
	float* __restrict__  hx1, float* __restrict__  hy1, float* __restrict__  hz1,
	float* __restrict__  hx2, float* __restrict__  hy2, float* __restrict__  hz2,
	float* __restrict__  alpha_, float alpha_mul,
	float* __restrict__  alpha1_, float alpha1_mul,
	float* __restrict__  alpha2_, float alpha2_mul,
	float* __restrict__  TCurie_, float TCurie_mul,
	float* __restrict__  Msat_, float Msat_mul,
	float* __restrict__  Msat1_, float Msat1_mul,
	float* __restrict__  Msat2_, float Msat2_mul,
	float* __restrict__  hth11x, float* __restrict__  hth11y, float* __restrict__  hth11z,
	float* __restrict__  hth21x, float* __restrict__  hth21y, float* __restrict__  hth21z,
	float* __restrict__  hth12x, float* __restrict__  hth12y, float* __restrict__  hth12z,
	float* __restrict__  hth22x, float* __restrict__  hth22y, float* __restrict__  hth22z,
	float* __restrict__  temp_, float temp_mul,
	float* __restrict__  x_, float x_mul,
	float* __restrict__  nv_, float nv_mul,
	float* __restrict__  mua_, float mua_mul,
	float* __restrict__  mub_, float mub_mul,
	float* __restrict__  J0aa_, float J0aa_mul,
	float* __restrict__  J0bb_, float J0bb_mul,
	float* __restrict__  J0ab_, float J0ab_mul,
  float* __restrict__  lambda0_, float lambda0_mul,
	int N) {

    const float kB=1.38064852e-23;

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
        float3 m1 = {mx1[i], my1[i], mz1[i]};
        float3 m2 = {mx2[i], my2[i], mz2[i]};
        float3 H1 = {hx1[i], hy1[i], hz1[i]};
        float3 H2 = {hx2[i], hy2[i], hz2[i]};
        float alpha = amul(alpha_, alpha_mul, i);
        float alphaa = amul(alpha1_, alpha1_mul, i);
        float alphab = amul(alpha2_, alpha2_mul, i);
        float TCurie = amul(TCurie_, TCurie_mul, i);
        float Msat = amul(Msat_, Msat_mul, i);
        float Msata = amul(Msat1_, Msat1_mul, i);
        float Msatb = amul(Msat2_, Msat2_mul, i);
        float x= amul(x_, x_mul, i);
        float nv = amul(nv_, nv_mul, i);
        float mua = amul(mua_, mua_mul, i);
        float mub = amul(mub_, mub_mul, i);
        float J0aa = amul(J0aa_, J0aa_mul, i)*nv*x;
        float J0bb = amul(J0bb_, J0bb_mul, i)*nv*(1.0f-x);
        float J0ab = amul(J0ab_, J0ab_mul, i)*(1.0f-x)*nv;
        float J0ba = amul(J0ab_, J0ab_mul, i)*x*nv;
        float temp = amul(temp_, temp_mul, i);
        float lambda0 = amul(lambda0_, lambda0_mul, i)*x*nv;

        if (temp==0) temp=0.0001; // to avoid zero division...

        float3 hth1a = {hth11x[i], hth11y[i],hth11z[i]};
        float3 hth2a = {hth21x[i], hth21y[i],hth21z[i]};
        float3 hth1b = {hth12x[i], hth12y[i],hth12z[i]};
        float3 hth2b = {hth22x[i], hth22y[i],hth22z[i]};
        float3 torquea;
        float3 torqueb;

        // Parametros de LLB

        float ma=sqrt(dot(m1,m1));
        float mb=sqrt(dot(m2,m2));
        if ((ma==0)||(mb==0))	{
					torquea = 0.0f*m1;
         	torqueb = 0.0f*m2;
				} else {

					float J01 = J0aa;
					float J02 = J0bb;
					float J012 = J0ab;
					float J021 = J0ba;
					TCurie = (J01+J02+pow(pow(J01-J02,2.0f)+4.0f*J012*J021,0.5f))/(6.0f*kB); // Eq (9) LowTempPhys 41 (9) 2015
					if ((fabs(temp-TCurie)<0.007*TCurie)&&(temp<TCurie)) {temp=0.993f*TCurie;}  // To avoid errors arround T=Tc
					if (temp>2.0*TCurie) temp=2.0*TCurie; // To avoid numerical problems much above TC

					/*float alphaparA;
					float alphaparB;
					float alphaperpA;
					float alphaperpB;
					if (temp<TCurie) {
						alphaparA=2.0f*alphaa*temp/(3.0f*TCurie);
						alphaparB=2.0f*alphab*temp/(3.0f*TCurie);
						alphaperpA=alphaa*(1.0f-temp/(3*TCurie));
						alphaperpB=alphab*(1.0f-temp/(3*TCurie));
					} else {
						alphaparA=2.0f*alphaa*temp/(3.0f*TCurie);
						alphaparB=2.0f*alphab*temp/(3.0f*TCurie);
						alphaperpA=alphaparA;
						alphaperpB=alphaparB;
					}*/

					float3 tolerancia;
					tolerancia.x=0*0.0001;
					tolerancia.y=0*0.0001;
					tolerancia.z=0*0.0001;
					float3 HMFA1=J0aa/mua*m1+J0ab/mua*m2+tolerancia;
					float3 HMFA2=J0bb/mub*m2+J0ba/mub*m1+tolerancia;

					float beta=1.0f/(kB*temp);
					float3 chiA=beta*mua*HMFA1;
					float3 chiB=beta*mub*HMFA2;
					float mchiA=sqrt(dot(chiA,chiA));
					float mchiB=sqrt(dot(chiB,chiB));
					float3 m0A;
					float3 m0B;
					float lambdaA;
					float lambdaB;
					float GammaApara;
					float GammaBpara;
					float GammaAperp;
					float GammaBperp;
					if (temp<TCurie/5){   // For Low Temp do not restict Langevin/Dlangevin small values
						m0A=dL0(mchiA)/mchiA*chiA;
						m0B=dL0(mchiB)/mchiB*chiB;
						lambdaA=2.0f*alphaa*kB*temp/mua;
						lambdaB=2.0f*alphab*kB*temp/mub;
						GammaApara=lambdaA*dL0(mchiA)/(mchiA*dLder0(mchiA));
						GammaBpara=lambdaB*dL0(mchiB)/(mchiB*dLder0(mchiB));
						GammaAperp=lambdaA/2.0f*(mchiA/dL0(mchiA)-1.0f);
						GammaBperp=lambdaB/2.0f*(mchiB/dL0(mchiB)-1.0f);
					} else {
						m0A=dL(mchiA)/mchiA*chiA;
						m0B=dL(mchiB)/mchiB*chiB;
						lambdaA=2.0f*alphaa*kB*temp/mua;
						lambdaB=2.0f*alphab*kB*temp/mub;
						GammaApara=lambdaA*dL(mchiA)/(mchiA*dLder(mchiA));
						GammaBpara=lambdaB*dL(mchiB)/(mchiB*dLder(mchiB));
						GammaAperp=lambdaA/2.0f*(mchiA/dL(mchiA)-1.0f);
						GammaBperp=lambdaB/2.0f*(mchiB/dL(mchiB)-1.0f);
					}

					//float3 Hi1=1/(beta*mua*dLder(mchiA))*(m1-m0A);
					//float3 Hi2=1/(beta*mub*dLder(mchiB))*(m2-m0B);
					//float3 h0=lambda0*(ma+(1-x)*mub/x/mua*mb)/ma/mb*((mua/mub)*Hi1-Hi2);
					//float3 hab = h0;
					//float3 hba = -1.0f*h0;


		// New Implementatiosn of Unai Non Equiibrium exchanges
		float alphaex;
		float alphaae=alphaa*2*dL(mchiA)/mchiA;
		float alphabe=alphab*2*dL(mchiB)/mchiB;
		alphaex=0.0*0.5f*(alphaae/x/ma+alphabe/(1.0f-x)/mb);
		float3 Ha=1/(beta*mua*dLder(mchiA))*(m1-m0A);
		float3 Hb=1/(beta*mub*dLder(mchiB))*(m2-m0B);
		// end of new parameters

		// Recuperamos todas las contribuciones
		H1=H1+J0aa/mua*m1+J0ab/mua*m2;
		H2=H2+J0bb/mub*m2+J0ba/mub*m1;

/*
		if (i==1) {
			printf("m0A;%e %e %e\n",m0A.x,m0A.y,m0A.z);
			printf("Hth;%e %e %e sc:%e T:%e\n",h_perp_scalea*hth1a.x,h_perp_scalea*hth1a.y,h_perp_scalea*hth1a.z,h_perp_scalea,temp);
		}
*/
    float3 m1xHmfa = cross(m1, H1);
    float3 m2xHmfa = cross(m2, H2);
    float m1dotm0a = dot(m1, m0A);
    float m2dotm0b = dot(m2, m0B);
    float3 m1xHtot1 = cross(m1, m0A);
    float3 m2xHtot2 = cross(m2, m0B);
    float3 m1xm1xHtot1 = cross(m1, m1xHtot1);
    float3 m2xm2xHtot2 = cross(m2, m2xHtot2);
    float gillba = 1.0f / (1.0f + alphaa * alphaa);
    float gillbb = 1.0f / (1.0f + alphab * alphab);


		// New Thermal Noise PRB 102 024428 (2020)

		float alphaparA=GammaApara/lambdaA;
		float alphaparB=GammaBpara/lambdaB;
		float alphaperA=GammaAperp/lambdaA;
		float alphaperB=GammaBperp/lambdaB;

		float3 v1A;
		v1A.x=m1.x*(m1.x +m1.y+m1.z)*hth1a.x;
		v1A.y=m1.y*(m1.x +m1.y+m1.z)*hth1a.y;
		v1A.z=m1.z*(m1.x +m1.y+m1.z)*hth1a.z;
		float3 v1B;
		v1B.x=m2.x*(m2.x +m2.y+m2.z)*hth1b.x;
		v1B.y=m2.y*(m2.x +m2.y+m2.z)*hth1b.y;
		v1B.z=m2.z*(m2.x +m2.y+m2.z)*hth1b.z;
		float3 v2A;
		v2A.x=ma-m1.x*(m1.x +m1.y+m1.z)*hth2a.x;
		v2A.y=ma-m1.y*(m1.x +m1.y+m1.z)*hth2a.y;
		v2A.z=ma-m1.z*(m1.x +m1.y+m1.z)*hth2a.z;
		float3 v2B;
		v2B.x=mb-m2.x*(m2.x +m2.y+m2.z)*hth2b.x;
		v2B.y=mb-m2.y*(m2.x +m2.y+m2.z)*hth2b.y;
		v2B.z=mb-m2.z*(m2.x +m2.y+m2.z)*hth2b.z;

		float D1A=0;
		float D1B=0;


		alphaparA=1;
		alphaparB=1;
		alphaperA=1;
		alphaperB=1;

		if (temp>TCurie) {
			D1A=sqrt(Msat/Msata/alphaparA/alpha)/ma/ma*(temp/TCurie-1.0f);
			D1B=sqrt(Msat/Msatb/alphaparB/alpha)/mb/mb*(temp/TCurie-1.0f);
		}
		float D2A=sqrt(Msat/Msata/alphaperA/alpha)/ma/ma;
		float D2B=sqrt(Msat/Msatb/alphaperB/alpha)/mb/mb;

/*		if (i==1) {
			printf("%e %e %e %e\n",alphaparA,alphaparB,alphaperA,alphaperB);
		}*/

		torquea = -gillba*(m1xHmfa+GammaApara*(1.0f-m1dotm0a/ma/ma)*m1+GammaAperp/ma/ma*(m1xm1xHtot1)+D1A*v1A+D2A*v2A)+gillba*alphaex*(mua*Ha-mub*Hb);
		torqueb = -gillbb*(m2xHmfa+GammaBpara*(1.0f-m2dotm0b/mb/mb)*m2+GammaBperp/mb/mb*(m2xm2xHtot2)+D1B*v1B+D2B*v2B)-gillbb*alphaex*(mua*Ha-mub*Hb);
    }
    t1x[i] = torquea.x;
    t1y[i] = torquea.y;
    t1z[i] = torquea.z;
    t2x[i] = torqueb.x;
    t2y[i] = torqueb.y;
    t2z[i] = torqueb.z;
    }
}