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
AF2TRenormBri(float* __restrict__  mx01, float* __restrict__  my01, float* __restrict__  mz01,
	float* __restrict__  mx02, float* __restrict__  my02, float* __restrict__  mz02,
float* __restrict__  mx1, float* __restrict__  my1, float* __restrict__  mz1,
	float* __restrict__  mx2, float* __restrict__  my2, float* __restrict__  mz2,
	float* __restrict__  alpha_, float alpha_mul,
	float* __restrict__  alpha1_, float alpha1_mul,
	float* __restrict__  alpha2_, float alpha2_mul,
	float* __restrict__  TCurie_, float TCurie_mul,
	float* __restrict__  Msat_, float Msat_mul,
	float* __restrict__  Msat1_, float Msat1_mul,
	float* __restrict__  Msat2_, float Msat2_mul,
	float* __restrict__  temp_,
	float* __restrict__  x_, float x_mul,
	float* __restrict__  nv_, float nv_mul,
	float* __restrict__  mua_, float mua_mul,
	float* __restrict__  mub_, float mub_mul,
	float* __restrict__  J0aa_, float J0aa_mul,
	float* __restrict__  J0bb_, float J0bb_mul,
	float* __restrict__  J0ab_, float J0ab_mul,
	float dt, float gammaA, float gammaB,
	float* __restrict__  JA_, float JA_mul,
	float* __restrict__  JB_, float JB_mul,
		int N) {

    const float kB=1.38064852e-23;
		float mmin=0.001; // Previously 0.01

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
        float3 m1 = {mx1[i], my1[i], mz1[i]};
        float3 m2 = {mx2[i], my2[i], mz2[i]};
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
				float JA = amul(JA_, JA_mul, i);
				float JB = amul(JB_, JB_mul, i);

        if (temp==0) temp=0.0001; // to avoid zero division...
        if (temp>2.0*TCurie) temp=2.0*TCurie; // To avoid numerical problems much above TC
		if (temp==TCurie) temp=TCurie-0.1;  // To avoid xpar and Bint divergence at T=Tc

        // Parametros de LLB

        float ma=sqrt(dot(m1,m1));
        float mb=sqrt(dot(m2,m2));

        if ((ma==0)||(mb==0))	{
 		} else {



			float J01 = J0aa;
			float J02 = J0bb;
			float J012 = J0ab;
			float J021 = J0ba;

			//if (i==1) printf("%e %e %e %e\n",J01,J02,J012,J021);
			//TCurie = (J01+J02+pow(pow(J01-J02,2.0f)+4.0f*J012*J021,0.5f))/(6.0f*kB); // Eq (9) LowTempPhys 41 (9) 2015
			if ((fabs(temp-TCurie)<0.007*TCurie)&&(temp<TCurie)) {temp=0.993f*TCurie;}  // To avoid errors arround T=Tc

			float mea;
			float meb;

	//		if (temp<TCurie) {

			// Parametros de LLB y calculo de m_e1,2

			// Calculo de m_e1,m_e2
			float A11, A12, A21, A22;
		 	float x10, x20;
			float x1, x2, y1, y2;
			float dx1, dx2;
			float dL1, dL2;
			float det, deti;
			float J11,J12,J21,J22;
			float m_e1,m_e2;
			float tolsq=1e-5;
			float tol=1e-4;

			float kB_T=kB*temp;
			float beta = 1.0/kB_T;

		   A11 = beta*J01;
		   A12 = beta*fabs(J012);
		   A21 = beta*fabs(J021);
		   A22 = beta*J02;

		   x1 = 0.8; x2 = 0.8;
		   y1 = Brillouin2(A11*x1+A12*x2,JA)-x1;
		   y2 = Brillouin2(A21*x1+A22*x2,JB)-x2;

		   //int iter=0;
		   do
		   {
		     x10 = x1; x20 = x2;
		     dL1 = DBrillouin2(A11*x1+A12*x2,JA);
		     dL2 = DBrillouin2(A21*x1+A22*x2,JB);
		     // Jacobian
		     J11 = A11*dL1-1;
		     J12 = A12*dL1;
		     J21 = A21*dL2;
		     J22 = A22*dL2-1;
		     det = J11*J22-J12*J21;
		     if(det == 0.0) {
		       // No more change. Calculate parameters with current x1, x2.
		       break;
		     }
		     deti = 1.0/det;
		     dx1 = -deti*(J22*y1-J12*y2);
		     dx2 = -deti*(-J21*y1+J11*y2);
		     x1 = x10 + dx1;
		     x2 = x20 + dx2;
		     y1 = Brillouin2(A11*x1+A12*x2,JA)-x1;
		     y2 = Brillouin2(A21*x1+A22*x2,JA)-x2;
		     //iter++;
		   } while( dx1*dx1>tolsq || dx2*dx2>tolsq );
		   m_e1 = x1>tol ? x1 : 0.0;
		   m_e2 = x2>tol ? x2 : 0.0;

			mea=m_e1;
			meb=m_e2;

			if (mea<mmin) {mea=mmin;}
			if (meb<mmin) {meb=mmin;}

	mx1[i]=mea/ma*m1.x;
	my1[i]=mea/ma*m1.y;
	mz1[i]=mea/ma*m1.z;
	mx2[i]=meb/mb*m2.x;
	my2[i]=meb/mb*m2.y;
	mz2[i]=meb/mb*m2.z;


	}


    }
}
