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
LLBtorqueAF2TBrillouin2(float* __restrict__  t1x, float* __restrict__  t1y, float* __restrict__  t1z,
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
	float* __restrict__  JA_, float JA_mul,
	float* __restrict__  JB_, float JB_mul,
	int N) {

    //const float kB=1.38064852e-23;
		//float mmin=0.01; // Previously 0.01

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
				//float xx=(i-N/2.0)/2000;
				//printf("%e %e %e\n",xx,Brillouin2(xx,0.5),DBrillouin2(xx,0.5));
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
				//float lambda0 = amul(lambda0_, lambda0_mul, i)*x*nv;
				float lambda0 = amul(lambda0_, lambda0_mul, i);
        float temp = temp_[i];
				float JA = amul(JA_, JA_mul, i);
				float JB = amul(JB_, JB_mul, i);

        float3 torquea;
        float3 torqueb;

				float ma=sqrt(dot(m1,m1));
				float mb=sqrt(dot(m2,m2));

				float3 hexa = -J0ab/(mua*ma*ma)*cross(m1, cross(m1, m2));
				float3 hexb = -J0ba/(mub*mb*mb)*cross(m2, cross(m2, m1));

				float3 hth1a = {hth11x[i], hth11y[i],hth11z[i]};
				float3 hth2a = {hth12x[i], hth12y[i],hth12z[i]};

				float h_par_scalea=sqrt(Msat/Msata*alphaa/alpha);
				float h_par_scaleb=sqrt(Msat/Msatb*alphab/alpha);

				H1=H1+hexa+h_par_scalea*hth1a;
				H2=H2+hexb+0.0*h_par_scalea*hth2a;

				//if (i==1) printf("%f %f %f %f %f %f %f\n",h_par_scalea,hth1a.x,hth1a.y,hth1a.z,H1.x, H1.y, H1.z);

				float gillba = 1.0f / (1.0f + alphaa * alphaa);
		    float gillbb = 1.0f / (1.0f + alphab * alphab);

        float3 m1xH1 = cross(m1, H1);
        torquea = gillba * (m1xH1 + alphaa * cross(m1, m1xH1));

				float3 m2xH2 = cross(m2, H2);
				torqueb = gillbb * (m2xH2 + alphab * cross(m2, m2xH2));

				t1x[i] = torquea.x;
				t1y[i] = torquea.y;
				t1z[i] = torquea.z;
				t2x[i] = torqueb.x;
				t2y[i] = torqueb.y;
				t2z[i] = torqueb.z;

    }
}
