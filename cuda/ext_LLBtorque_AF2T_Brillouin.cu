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
LLBtorqueAF2TBrillouin(float* __restrict__  t1x, float* __restrict__  t1y, float* __restrict__  t1z,
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

    const float kB=1.38064852e-23;
		float mmin=0.01; // Previously 0.01

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

        if (temp==0) temp=0.0001; // to avoid zero division...
        if (temp>2.0*TCurie) temp=2.0*TCurie; // To avoid numerical problems much above TC
		if (temp==TCurie) temp=TCurie-0.1;  // To avoid xpar and Bint divergence at T=Tc

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

//		if (i==1) printf("%e %e %e %e %e\n",temp,mea,meb,JA,JB);
/*		if (i==1) printf("m1 %e %e %e\n",m1.x,m1.y,m1.z);
		if (i==1) printf("m2 %e %e %e\n",m2.x,m2.y,m2.z);
		if (i==1) printf("Cosas %e %e %e %e %e %e\n",H1.x,H1.y,H1.z,H2.x,H2.y,H2.z);
*/
		float chiA=(J0aa*mea+fabs(J0ab)*meb)/(kB*temp);  // Arguments of Langevin functions Error in PRB 054401 using Low Temp Phys
		float chiB=(J0bb*meb+fabs(J0ba)*mea)/(kB*temp);

		float xpara;
		float xparb;

//		Following notation of LowTemp Phys (eq 13), to avoid numerical errors float<1e-38
		if (temp<TCurie) {
			xpara=(mub/(kB*temp)*DBrillouin2(chiA,JA)*fabs(J0ab)/(kB*temp)*DBrillouin2(chiB,JB)+mua/(kB*temp)*DBrillouin2(chiA,JA)*(1.0f-J0bb/(kB*temp)*DBrillouin2(chiB,JB)))/((1.0f-J0aa/(kB*temp)*DBrillouin2(chiA,JA))*(1.0f-J0bb/(kB*temp)*DBrillouin2(chiB,JB))-fabs(J0ba)/(kB*temp)*DBrillouin2(chiA,JA)*fabs(J0ab)/(kB*temp)*DBrillouin2(chiB,JB));
			xparb=(mua/(kB*temp)*DBrillouin2(chiB,JB)*fabs(J0ba)/(kB*temp)*DBrillouin2(chiA,JA)+mub/(kB*temp)*DBrillouin2(chiB,JB)*(1.0f-J0aa/(kB*temp)*DBrillouin2(chiA,JA)))/((1.0f-J0bb/(kB*temp)*DBrillouin2(chiB,JB))*(1.0f-J0aa/(kB*temp)*DBrillouin2(chiA,JA))-fabs(J0ab)/(kB*temp)*DBrillouin2(chiB,JB)*fabs(J0ba)/(kB*temp)*DBrillouin2(chiA,JA));
		} else { // T>Tc Lder=1/3
			xpara=(mub/(kB*temp)/3.0f*fabs(J0ab)/(kB*temp)/3.0f+mua/(kB*temp)/3.0f*(1.0f-J0bb/(kB*temp)/3.0f))/((1.0f-J0aa/(kB*temp)/3.0f)*(1.0f-J0bb/(kB*temp)/3.0f)-fabs(J0ba)/(kB*temp)/3.0f*fabs(J0ab)/(kB*temp)/3.0f);
			xparb=(mua/(kB*temp)/3.0f*fabs(J0ba)/(kB*temp)/3.0f+mub/(kB*temp)/3.0f*(1.0f-J0aa/(kB*temp)/3.0f))/((1.0f-J0bb/(kB*temp)/3.0f)*(1.0f-J0aa/(kB*temp)/3.0f)-fabs(J0ab)/(kB*temp)/3.0f*fabs(J0ba)/(kB*temp)/3.0f);
		}
/*
// Alternate method for not knowing Tc
		if (mea>mmin) {
			xpara=(mub/(kB*temp)*DBrillouin2(chiA,JA)*fabs(J0ab)/(kB*temp)*DBrillouin2(chiB,JB)+mua/(kB*temp)*DBrillouin2(chiA,JA)*(1.0f-J0bb/(kB*temp)*DBrillouin2(chiB,JB)))/((1.0f-J0aa/(kB*temp)*DBrillouin2(chiA,JA))*(1.0f-J0bb/(kB*temp)*DBrillouin2(chiB,JB))-fabs(J0ba)/(kB*temp)*DBrillouin2(chiA,JA)*fabs(J0ab)/(kB*temp)*DBrillouin2(chiB,JB));
		} else {
			xpara=(mub/(kB*temp)/3.0f*fabs(J0ab)/(kB*temp)/3.0f+mua/(kB*temp)/3.0f*(1.0f-J0bb/(kB*temp)/3.0f))/((1.0f-J0aa/(kB*temp)/3.0f)*(1.0f-J0bb/(kB*temp)/3.0f)-fabs(J0ba)/(kB*temp)/3.0f*fabs(J0ab)/(kB*temp)/3.0f);
		}

		if (meb>mmin) {
			xparb=(mua/(kB*temp)*DBrillouin2(chiB,JB)*fabs(J0ba)/(kB*temp)*DBrillouin2(chiA,JA)+mub/(kB*temp)*DBrillouin2(chiB,JB)*(1.0f-J0aa/(kB*temp)*DBrillouin2(chiA,JA)))/((1.0f-J0bb/(kB*temp)*DBrillouin2(chiB,JB))*(1.0f-J0aa/(kB*temp)*DBrillouin2(chiA,JA))-fabs(J0ab)/(kB*temp)*DBrillouin2(chiB,JB)*fabs(J0ba)/(kB*temp)*DBrillouin2(chiA,JA));
		} else {
			xparb=(mua/(kB*temp)/3.0f*fabs(J0ba)/(kB*temp)/3.0f+mub/(kB*temp)/3.0f*(1.0f-J0aa/(kB*temp)/3.0f))/((1.0f-J0bb/(kB*temp)/3.0f)*(1.0f-J0aa/(kB*temp)/3.0f)-fabs(J0ab)/(kB*temp)/3.0f*fabs(J0ba)/(kB*temp)/3.0f);
		}
// End alterate
*/
		if (xpara<1e-5) xpara=1e-5;
		if (xparb<1e-5) xparb=1e-5;
		//if (i==1) printf("%e %e\n",xpara,xparb);



		float3 hexa = -J0ab/(mua*ma*ma)*cross(m1, cross(m1, m2));
		float3 hexb = -J0ba/(mub*mb*mb)*cross(m2, cross(m2, m1));

		float tauA=fabs(dot(m2,m1))/mb;
		float tauB=fabs(dot(m1,m2))/ma;
		float taueA=tauA*mea/ma;
		float taueB=tauB*meb/mb;

		float lambdaA=(1.0f/xpara+fabs(J0ab)/mua*xparb/xpara);
		float lambdaB=(1.0f/xparb+fabs(J0ba)/mub*xpara/xparb);



		//if (i==1) printf("%e %e\n",lambdaA,lambdaB);
		//if ((lambdaA>1e3)||(lambdaA!=lambdaA)) lambdaA=1e3;
		//if ((lambdaB>1e3)||(lambdaB!=lambdaB)) lambdaB=1e3;

		//if (lambdaA<100) lambdaA=100.0f;
		//if (lambdaB<100) lambdaB=100.0f;

    float3 heffa;
		float3 heffb;
    if (temp>TCurie) {
			//heffa = -(1.0f/xpara+fabs(J0ab)/mua*xparb/xpara)*m1+fabs(J0ab)/mua*tauB/ma*m1;
      //heffb = -(1.0f/xparb+fabs(J0ba)/mub*xpara/xparb)*m2+fabs(J0ba)/mub*tauA/mb*m2;
		  heffa = -lambdaA*m1+fabs(J0ab)/mua*tauB/ma*m1;
      heffb = -lambdaB*m2+fabs(J0ba)/mub*tauA/mb*m2;
    } else {
      heffa = -lambdaA*((ma-mea)/mea)*m1+fabs(J0ab)/mua*(tauB-taueB)/mea*m1;
      heffb = -lambdaB*((mb-meb)/meb)*m2+fabs(J0ba)/mub*(tauA-taueA)/meb*m2;
    }
/*
// Alternate method for not knowing Tc
if (mea<mmin) {
	heffa = -lambdaA*m1+fabs(J0ab)/mua*tauB/ma*m1;
} else {
	heffa = -lambdaA*((ma-mea)/mea)*m1+fabs(J0ab)/mua*(tauB-taueB)/mea*m1;
}

if (meb<mmin) {
	heffb = -lambdaB*m2+fabs(J0ba)/mub*tauA/mb*m2;
} else {
	heffb = -lambdaB*((mb-meb)/meb)*m2+fabs(J0ba)/mub*(tauA-taueA)/meb*m2;
}
// end alternate
*/

		float alphaparA;
		float alphaparB;
		float alphaperpA;
		float alphaperpB;
		if (temp<TCurie) {
			alphaparA=2.0f*alphaa*kB*temp*mea/(J0aa*mea+fabs(J0ab)*meb);
			alphaparB=2.0f*alphab*kB*temp*meb/(J0bb*meb+fabs(J0ba)*mea);
			alphaperpA=alphaa*(1.0f-kB*temp*mea/(J0aa*mea+fabs(J0ab)*meb));
			alphaperpB=alphab*(1.0f-kB*temp*meb/(J0bb*meb+fabs(J0ba)*mea));
		} else {
			alphaparA=2.0f*alphaa*temp/(3.0f*TCurie);
			alphaparB=2.0f*alphab*temp/(3.0f*TCurie);
			alphaperpA=alphaparA;
			alphaperpB=alphaparB;
		}

// Alternate method for not knowing Tc
/*
if (mea>mmin) {
	alphaparA=2.0f*alphaa*kB*temp*mea/(J0aa*mea+fabs(J0ab)*meb);
	alphaperpA=alphaa*(1.0f-kB*temp*mea/(J0aa*mea+fabs(J0ab)*meb));
} else {
	alphaparA=2.0f*alphaa*temp/(3.0f*TCurie);
	alphaperpA=alphaparA;
}

if (meb>mmin) {
	alphaparB=2.0f*alphab*kB*temp*meb/(J0bb*meb+fabs(J0ba)*mea);
	alphaperpB=alphab*(1.0f-kB*temp*meb/(J0bb*meb+fabs(J0ba)*mea));
} else {
	alphaparB=2.0f*alphab*temp/(3.0f*TCurie);
	alphaperpB=alphaparB;
}


// End alternate
*/

		float h_perp_scalea=sqrt(Msat/Msata*(alphaperpA-alphaparA)/(alpha*alphaperpA*alphaperpA));
		float h_perp_scaleb=sqrt(Msat/Msatb*(alphaperpB-alphaparB)/(alpha*alphaperpB*alphaperpB));
		float h_par_scalea=sqrt(Msat/Msata*alphaparA/alpha);
		float h_par_scaleb=sqrt(Msat/Msatb*alphaparB/alpha);

		// Unai Non Equilibrium Exchange between lattices
		float3 Ha;
		float3 Hb;
		float3 hab;
		float3 hba;
		float alphaex;
		// Old implementation of Unai Non Equilibrium exchanges

    if (lambda0>0) {
		float3 h0=lambda0*(ma+(1-x)*mub/x/mua*mb)/ma/mb*((mua/mub)*heffa-heffb);
		alphaex=lambda0;
		Ha=0.0*h0;
		Hb=0.0*h0;
		hab = h0;
		hba = -1.0*h0;
		H1=H1+hab;
		H2=H2+hba;
	  } else {
		// New Implementation of Unai Non Equilibrium exchanges
		float beta=1.0f/(kB*temp);
		float3 H1mfa=H1+J0aa/mua*m1+J0ab/mua*m2;
		float3 H2mfa=H2+J0bb/mub*m2+J0ba/mub*m1;
		float3 chiAmfa=beta*mua*H1mfa;
		float3 chiBmfa=beta*mub*H2mfa;
		float mchiAmfa=sqrt(dot(chiAmfa,chiAmfa));
		float mchiBmfa=sqrt(dot(chiBmfa,chiBmfa));
		float alphaae=alphaa*2*DBrillouin2(mchiAmfa,JA)/mchiAmfa;
		float alphabe=alphab*2*DBrillouin2(mchiBmfa,JB)/mchiBmfa;
		alphaex=2.8f*0.5f*(alphaae/nv/x/ma+alphabe/(1.0f-x)/nv/mb);
		if (alphaex>0.1f) alphaex=0.1f; //0.1
		Ha=-1.0f*heffa;
		Hb=-1.0f*heffb;
	  }

		H1=H1+heffa+hexa;
		H2=H2+heffb+hexb;

    float3 htot1=H1+h_perp_scalea*hth1a;
    float3 htot2=H2+h_perp_scaleb*hth1b;

    float3 m1xH1 = cross(m1, H1);
    float3 m2xH2 = cross(m2, H2);
    float m1dotH1 = dot(m1, H1);
    float m2dotH2 = dot(m2, H2);
    float3 m1xHtot1 = cross(m1, htot1);
    float3 m2xHtot2 = cross(m2, htot2);
    float3 m1xm1xHtot1 = cross(m1, m1xHtot1);
    float3 m2xm2xHtot2 = cross(m2, m2xHtot2);
    float gillba = 1.0f / (1.0f + alphaperpA * alphaperpA);
    float gillbb = 1.0f / (1.0f + alphaperpB * alphaperpB);

		// New exchange Unai (last term or the addition to H1)
		torquea = -gillba*m1xH1+gillba*alphaparA/ma/ma*m1dotH1*m1-gillba*alphaperpA/ma/ma*(m1xm1xHtot1)+h_par_scalea*hth2a-alphaex*(Ha-Hb);
    torqueb = -gillbb*m2xH2+gillbb*alphaparB/mb/mb*m2dotH2*m2-gillbb*alphaperpB/mb/mb*(m2xm2xHtot2)+h_par_scaleb*hth2b+alphaex*(Ha-Hb);

		}

    t1x[i] = torquea.x;
    t1y[i] = torquea.y;
    t1z[i] = torquea.z;
    t2x[i] = torqueb.x;
    t2y[i] = torqueb.y;
    t2z[i] = torqueb.z;
    }
}
