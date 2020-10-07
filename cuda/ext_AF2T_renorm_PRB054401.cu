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
AF2TRenorm(float* __restrict__  mx01, float* __restrict__  my01, float* __restrict__  mz01,
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
	int N) {

    const float kB=1.38064852e-23;

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
        float3 m1 = {mx1[i], my1[i], mz1[i]};
        float3 m2 = {mx2[i], my2[i], mz2[i]};
        float3 m01 = {mx01[i], my01[i], mz01[i]};
        float3 m02 = {mx02[i], my02[i], mz02[i]};
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

        if (temp==0) temp=0.0001; // to avoid zero division...
        if (temp>2.0*TCurie) temp=2.0*TCurie; // To avoid numerical problems much above TC
		if (temp==TCurie) temp=TCurie-0.1;  // To avoid xpar and Bint divergence at T=Tc

        // Parametros de LLB

        float ma=sqrt(dot(m1,m1));
        float mb=sqrt(dot(m2,m2));
        float ma0=sqrt(dot(m01,m01));
        float mb0=sqrt(dot(m02,m02));
        if ((ma==0)||(mb==0))	{
 		} else {

		float J01 = J0aa;
		float J02 = J0bb;
		float J012 = J0ab;
		float J021 = J0ba;
		TCurie = (J01+J02+pow(pow(J01-J02,2.0f)+4.0f*J012*J021,0.5f))/(6.0f*kB); // Eq (9) LowTempPhys 41 (9) 2015
		if ((fabs(temp-TCurie)<0.007*TCurie)&&(temp<TCurie)) {temp=0.993f*TCurie;}  // To avoid errors arround T=Tc

		float mea;
		float meb;

		if (temp<TCurie) {

		// Parametros de LLB y calculo de m_e1,2

		float Told=temp;
		bool lin=false;   // Linear dependence at low temperatures due to numerical inestabilities
		if (temp<TCurie/10.0f) {
			Told=temp;
			lin=true;
			temp=TCurie/10.0f;
		}
		bool lin2=false;   // Linear dependence at close to Tc due to numerical inestabilities
		if (temp>TCurie*0.995) {
			Told=temp;
			lin2=true;
			temp=TCurie*0.995;
		}

		// Calculo de m_e1,m_e2
		float A11, A12, A21, A22;
	 	float x10, x20;
		float x1, x2, y1, y2;
		float dx1, dx2;
		float dL1, dL2;
		float det, deti;
		float J11,J12,J21,J22;
		float m_e1,m_e2;
		float tolsq=1e-4;
		float tol=1e-4;
		float kB_T=kB*temp;
		float beta = 1.0/kB_T;
		if(kB_T == 0) {
	    	m_e1=1.0;
	    	m_e2=1.0;
		}
		A11 = beta*J01;
		A12 = beta*fabs(J012);
		A21 = beta*fabs(J021);
		A22 = beta*J02;
		x1 = 0.9; x2 = 0.9;
		y1 = Langevin(A11*x1+A12*x2)-x1;
		y2 = Langevin(A21*x1+A22*x2)-x2;
		do {
    		x10 = x1; x20 = x2;
			dL1 = LangevinDeriv(A11*x1+A12*x2);
			dL2 = LangevinDeriv(A21*x1+A22*x2);
			// Jacobian
			J11 = A11*dL1-1.0f;
			J12 = A12*dL1;
			J21 = A21*dL2;
			J22 = A22*dL2-1.0f;
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
			y1 = Langevin(A11*x1+A12*x2)-x1;
			y2 = Langevin(A21*x1+A22*x2)-x2;
		} while( dx1*dx1>tolsq || dx2*dx2>tolsq );
		m_e1 = x1>tol ? x1 : 0.001;
		m_e2 = x2>tol ? x2 : 0.001;
		mea=m_e1;
		meb=m_e2;
		if (lin==true) {
			mea=1.0f-(mea-1.0f)/(-temp)*(Told);
			meb=1.0f-(meb-1.0f)/(-temp)*(Told);
			temp=Told;
		}
		if (lin2==1) {
			mea=mea*(TCurie-Told)/(0.005f*TCurie);
			meb=meb*(TCurie-Told)/(0.005f*TCurie);
			temp=Told;
		}
		} else	{
			mea=0.01f;
			meb=0.01f;
		}
		if (mea<0.01f) {mea=0.01f;}
		if (meb<0.01f) {meb=0.01f;}

		float chiA=(J0aa*mea+fabs(J0ab)*meb)/(kB*temp);  // Arguments of Langevin functions Error in PRB 054401 using Low Temp Phys
		float chiB=(J0bb*meb+fabs(J0ba)*mea)/(kB*temp);

		float xpara;
		float xparb;

//		Following notation of LowTemp Phys (eq 13), to avoid numerical errors float<1e-38
		if (temp<TCurie) {
			xpara=(mub/(kB*temp)*Lder(chiA)*fabs(J0ab)/(kB*temp)*Lder(chiB)+mua/(kB*temp)*Lder(chiA)*(1.0f-J0bb/(kB*temp)*Lder(chiB)))/((1.0f-J0aa/(kB*temp)*Lder(chiA))*(1.0f-J0bb/(kB*temp)*Lder(chiB))-fabs(J0ba)/(kB*temp)*Lder(chiA)*fabs(J0ab)/(kB*temp)*Lder(chiB));
			xparb=(mua/(kB*temp)*Lder(chiB)*fabs(J0ba)/(kB*temp)*Lder(chiA)+mub/(kB*temp)*Lder(chiB)*(1.0f-J0aa/(kB*temp)*Lder(chiA)))/((1.0f-J0bb/(kB*temp)*Lder(chiB))*(1.0f-J0aa/(kB*temp)*Lder(chiA))-fabs(J0ab)/(kB*temp)*Lder(chiB)*fabs(J0ba)/(kB*temp)*Lder(chiA));
		} else { // T>Tc Lder=1/3
			xpara=(mub/(kB*temp)/3.0f*fabs(J0ab)/(kB*temp)/3.0f+mua/(kB*temp)/3.0f*(1.0f-J0bb/(kB*temp)/3.0f))/((1.0f-J0aa/(kB*temp)/3.0f)*(1.0f-J0bb/(kB*temp)/3.0f)-fabs(J0ba)/(kB*temp)/3.0f*fabs(J0ab)/(kB*temp)/3.0f);
			xparb=(mua/(kB*temp)/3.0f*fabs(J0ba)/(kB*temp)/3.0f+mub/(kB*temp)/3.0f*(1.0f-J0aa/(kB*temp)/3.0f))/((1.0f-J0bb/(kB*temp)/3.0f)*(1.0f-J0aa/(kB*temp)/3.0f)-fabs(J0ab)/(kB*temp)/3.0f*fabs(J0ba)/(kB*temp)/3.0f);
		}

		float tauA=fabs(dot(m2,m1))/mb;
		float tauB=fabs(dot(m1,m2))/ma;

		float lambdaA=(1.0f/xpara+fabs(J0ab)/mua*xparb/xpara);
		float lambdaB=(1.0f/xparb+fabs(J0ba)/mub*xpara/xparb);

		float alphaparA;
		float alphaparB;

		if (temp<TCurie) {
			alphaparA=2.0f*alphaa*kB*temp*mea/(J0aa*mea+fabs(J0ab)*meb);
			alphaparB=2.0f*alphab*kB*temp*meb/(J0bb*meb+fabs(J0ba)*mea);
		} else {
			alphaparA=2.0f*alphaa*temp/(3.0f*TCurie);
			alphaparB=2.0f*alphab*temp/(3.0f*TCurie);

		}
	tauA=1.0f/(lambdaA*alphaparA*gammaA);
	tauB=1.0f/(lambdaB*alphaparB*gammaB);

	//if (i==1) printf("%e %e %e %e %e %e\n",tauA,tauB,ma,mea,1.0f/ma* ( mea+(ma-mea)*exp(-dt/tauA) ),dt);

	m1=1.0f/ma* ( mea+(ma0-mea)*exp(-dt/tauA) )*m1;
	m2=1.0f/mb* ( meb+(mb0-meb)*exp(-dt/tauB) )*m2;
	mx1[i]=m1.x;
	my1[i]=m1.y;
	mz1[i]=m1.z;
	mx2[i]=m2.x;
	my2[i]=m2.y;
	mz2[i]=m2.z;


	}


    }
}
