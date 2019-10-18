#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "ext_LLB_functions.h"

// Landau-Lifshitz torque. JPhysD Unai 2017 Using Langevin instead of Brillouin
extern "C" 

__global__ void
LLBtorque4JH(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__  hx, float* __restrict__  hy, float* __restrict__  hz,
          float* __restrict__  alpha_, float alpha_mul,
          float* __restrict__  TCurie_, float TCurie_mul,
          float* __restrict__  Msat_, float Msat_mul,
	  float* __restrict__  hth1x, float* __restrict__  hth1y, float* __restrict__  hth1z,
	  float* __restrict__  hth2x, float* __restrict__  hth2y, float* __restrict__  hth2z,
          float* __restrict__  tempJH,
	  float* __restrict__  a1_, float a1_mul,
          int N) {

    const float MU0=1.2566370614e-6;
    const float MUB=9.27400949e-24;
    const float kB=1.38e-23;
//    const float GammaLL=1.7595e11;
  
    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {
        float3 m = {mx[i], my[i], mz[i]};
        float3 H = {hx[i], hy[i], hz[i]};
        float alpha = amul(alpha_, alpha_mul, i);    // Do not understand what is this but seems to work this way
        float TCurie = amul(TCurie_, TCurie_mul, i);
        float Msat = amul(Msat_, Msat_mul, i);
        float temp = tempJH[i]; 
        if (temp==0) temp=0.0001; // to avoid zero division...
        if (temp>2.0*TCurie) temp=2.0*TCurie; // To avoid numerical problems much above TC
	if (temp==TCurie) temp=TCurie-0.1;  // To avoid xpar and Bint divergence at T=Tc
        float3 hth1 = {hth1x[i], hth1y[i],hth1z[i]};
        float3 hth2 = {hth2x[i], hth2y[i],hth2z[i]};
        float3 torque;

        // Parametros de LLB
        float alphapar;
        float alphaperp;
        float3 Bint;
        float me;
        float xpar;
        float m2;
	
        m2=dot(m,m);
	if (m2==0) torque = 0.0f*m;
 
        if ((m2!=0)&&(TCurie!=0))
        {
         alphapar=alpha*2.0f*temp/(3.0f*TCurie);
 	        
	 if (temp<=TCurie)  // T<Tc
         {
	  if (a1==0) a1=1.0f;
	  float t_Tc=pow(temp/TCurie,a1);
          //me=pow(1.0f-pow(temp/TCurie,1.23329f),0.43392f);
	  me=pow(1.0f-pow(t_Tc,1.23329f),0.43392f);
          if (me<0.001) me=0.001; //avoid zero division and numerical inestabilities close to Tc
          float beta=1.0f/(kB*(temp));
          //float argdB=3*TCurie/temp*me;   // me not Ms
          float argdB=3/t_Tc*me;   // me not Ms
          float dB=1.0f/(argdB*argdB)-1.0f/(sinh(argdB)*sinh(argdB));
          //xpar=(1.4f*MUB*MU0*beta)*(dB/(1.0f-3.0f*TCurie/(temp)*dB));
          xpar=(1.4f*MUB*MU0*beta)*(dB/(1.0f-3.0f/t_Tc*dB));

          if (xpar>1e-6) xpar=1e-6;
          if (xpar<8e-11) xpar=8e-11f;
          //alphaperp=alpha*(1.0f-temp/(3.0f*TCurie));
          alphaperp=alpha*(1.0f-t_Tc/3.0f);
          Bint=MU0*(1.0f/(2.0f*xpar)*(1.0f-m2/(me*me)))*m;
         }
         else        //T>Tc
         {
         xpar=0.2f*(1.4f*MUB*MU0/kB)*1.0f/(temp-TCurie);

//	 if (xpar==0) xpar=0.00001;
	 if (xpar>1e-6) xpar=1e-6;
	 if (xpar<8e-11) xpar=8e-11;

         alphaperp=alphapar;

         Bint=MU0*(-1.0f/xpar*(1.0f+3.0f/5.0f*m2*(TCurie/(temp-TCurie))))*m;
         };

         // LLB vector operations

  	 float h_perp_scale=sqrt((alphaperp-alphapar)/(alpha*alphaperp*alphaperp));
  	 float h_par_scale=sqrt(alphapar/alpha);
  
         H=H+Bint;
         float3 htot=H+h_perp_scale*hth1;

         float3 mxH = cross(m, H);
         float mdotH = dot(m, H);
         float3 mxHtot = cross(m, htot);
         float3 mxmxHtot = cross(m, mxHtot);
         float gillb = 1.0f / (1.0f + alpha * alpha);
 
         torque = -gillb*mxH+gillb*alphapar/m2*mdotH*m-gillb*alphaperp/m2*(mxmxHtot)+h_par_scale*hth2;
         }

        tx[i] = torque.x;
        ty[i] = torque.y;
        tz[i] = torque.z;

    }
}
