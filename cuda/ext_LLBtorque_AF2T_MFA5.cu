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
LLBtorqueAF2TMFA5(float* __restrict__  t1x, float* __restrict__  t1y, float* __restrict__  t1z,
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
	float* __restrict__  temp_,
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
        float temp = temp_[i];
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
				if (ma<0.001) ma=0.001;
				if (mb<0.001) mb=0.001;
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

					float alphaparA;
					float alphaparB;
					float alphaperpA;
					float alphaperpB;
					alphaparA=2.0f*alphaa*temp/(3.0f*TCurie);
					alphaparB=2.0f*alphab*temp/(3.0f*TCurie);
					if (temp<TCurie) {
						alphaperpA=alphaa*(1.0f-temp/(3*TCurie));
						alphaperpB=alphab*(1.0f-temp/(3*TCurie));
					} else {
						alphaperpA=alphaparA;
						alphaperpB=alphaparB;
					}


					float3 HMFA1=J0aa/mua*m1+J0ab/mua*m2+0*H1;
					float3 HMFA2=J0bb/mub*m2+J0ba/mub*m1+0*H2;

					float beta=1.0f/(kB*temp);
					float3 chiA3=beta*mua*HMFA1;
					float3 chiB3=beta*mub*HMFA2;
					float chiA=sqrt(dot(chiA3,chiA3));
					float chiB=sqrt(dot(chiB3,chiB3));


		float h_perp_scalea=sqrt(Msat/Msata*(alphaperpA-alphaparA)/(alpha*alphaperpA*alphaperpA));
		float h_perp_scaleb=sqrt(Msat/Msatb*(alphaperpB-alphaparB)/(alpha*alphaperpB*alphaperpB));
		float h_par_scalea=0*sqrt(Msat/Msata*alphaparA/alpha);
		float h_par_scaleb=0*sqrt(Msat/Msatb*alphaparB/alpha);

//		printf("%e %e %e %e \n",h_perp_scalea,h_perp_scaleb,h_par_scalea,h_par_scaleb);
//		printf("%e %e %e %e \n",temp,hth1a.z,hth1a.y,hth1a.z);

		// Recuperamos todas las contribuciones
		H1=H1-J0ab/mua*cross(m1,cross(m1,m2));
		H2=H2-J0bb/mub*cross(m2,cross(m2,m1));

		float3 htotA=H1+h_perp_scalea*hth1a;
    float3 htotB=H2+h_perp_scaleb*hth1b;

		float LA=NL(chiA);
		float LB=NL(chiB);
		if (LA<0.01) {LA=0.01;}
		if (LB<0.01) {LB=0.01;}
		float HintA=(1.0f-(LA)/ma)/(mua*beta*DNL(chiA));
		float HintB=(1.0f-(LB)/mb)/(mub*beta*DNL(chiB));

		// Old Unai Non Equilibrium Exchange between lattices
		float3 hab;
		float3 hba;
		float3 h0=lambda0*(ma+(1-x)*mub/x/mua*mb)/ma/mb*((mua/mub)*HintA*m1-HintB*m2);
		hab = -h0;
		hba = h0;
		H1=H1+hab;
		H2=H2+hba;

    float3 m1xH1 = cross(m1, H1);
    float3 m2xH2 = cross(m2, H2);
    float m1dotH1 = dot(m1, H1);
    float m2dotH2 = dot(m2, H2);
    float3 m1xHtot1 = cross(m1, htotA);
    float3 m2xHtot2 = cross(m2, htotB);
    float3 m1xm1xHtot1 = cross(m1, m1xHtot1);
    float3 m2xm2xHtot2 = cross(m2, m2xHtot2);
    float gillba = 1.0f / (1.0f + alphaa * alphaa);
    float gillbb = 1.0f / (1.0f + alphab * alphab);

		torquea = gillba*(m1xH1-alphaparA*(HintA-m1dotH1/(ma*ma))*m1-alphaperpA/(ma*ma)*m1xm1xHtot1)+h_par_scalea*hth2a;
		torqueb = gillbb*(m2xH2-alphaparB*(HintB-m2dotH2/(mb*mb))*m2-alphaperpB/(mb*mb)*m2xm2xHtot2)+h_par_scaleb*hth2b;

    }
    t1x[i] = torquea.x;
    t1y[i] = torquea.y;
    t1z[i] = torquea.z;
    t2x[i] = torqueb.x;
    t2y[i] = torqueb.y;
    t2z[i] = torqueb.z;
    }
}
