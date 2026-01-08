#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"
#include "exchange.h"

extern "C"

 __global__ void
evaldt02TBC2(float* __restrict__  tempe_,      float* __restrict__ dt0e_,
	float* __restrict__  templ_,      float* __restrict__ dt0l_,
	float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
//                float* __restrict__ Ke_, float Ke_mul,
                float* __restrict__ Ke_,
                float* __restrict__ Ce_, float Ce_mul,
//                float* __restrict__ Kl_, float Kl_mul,
                float* __restrict__ Kl_,
                float* __restrict__ Cl_, float Cl_mul,
                float* __restrict__ Gel_, float Gel_mul,
                float* __restrict__ Dth_, float Dth_mul,
                float* __restrict__ Tsubsth_, float Tsubsth_mul,
                float* __restrict__ Tausubsth_, float Tausubsth_mul,
                float* __restrict__ res_, float res_mul,
                float* __restrict__ Qext_, float Qext_mul,
	              float* __restrict__ cdx_, float cdx_mul,
                float* __restrict__ cdy_, float cdy_mul,
                float* __restrict__ cdz_, float cdz_mul,
                float* __restrict__ jx_, float jx_mul,
                float* __restrict__ jy_, float jy_mul,
                float* __restrict__ jz_, float jz_mul,
            		float wx, float wy, float wz, int Nx, int Ny, int Nz,
                float* __restrict__ vol,
                uint8_t* __restrict__ regions,
                float scaletausubs
                ) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    // central cell
    int i = idx(ix, iy, iz);
    uint8_t r0 = regions[i];
	  float mm = (vol == NULL? 1.0f: vol[i]);
    float3 m0={mx[i], my[i], mz[i]};
    dt0e_[i]=0.0;
    dt0l_[i]=0.0;

    if (mm!=0)
    {

    float3 J = vmul(jx_, jy_, jz_, jx_mul, jy_mul, jz_mul, i);
    //float Ke = amul(Ke_, Ke_mul, i);
    float Ce = amul(Ce_, Ce_mul, i);
    //float Kl = amul(Kl_, Kl_mul, i);
    float Cl = amul(Cl_, Cl_mul, i);
    float Dth = amul(Dth_, Dth_mul, i);
    float Gel = amul(Gel_, Gel_mul, i);
    float3 cd = vmul(cdx_, cdy_, cdz_, cdx_mul, cdy_mul, cdz_mul, i);

    float Tsubsth = amul(Tsubsth_, Tsubsth_mul, i);
    float Tausubsth = amul(Tausubsth_, Tausubsth_mul, i);
    float res = amul(res_, res_mul, i);
    float Qext = amul(Qext_, Qext_mul, i);

    float Te = tempe_[i];
    if (Te!=0) {Ce=Ce*Te/300.0;} else {Ce=Ce*1.0/300.0;}  // To account for temperature dependence. Cl is almost constat at T>TD
    float Tl = templ_[i];

    //int i_;    // neighbor index
    //float mm_;

    //float tempve=0;
    //float tempvl=0;
    float ke = Ke_[symidx(r0, r0)];
    float kl = Kl_[symidx(r0, r0)];

    float lapTex=0,lapTey=0,lapTez=0;
    float lapTlx=0,lapTly=0,lapTlz=0;
    int il, ir,rl, rr;
    float Ter,Tel, Tlr,Tll, kel, ker, kll, klr, ml, mr;


    // x derivatives
    il=idx(ix-1, iy, iz);
    ir=idx(ix+1, iy, iz);
    rl=regions[il];
    rr=regions[ir];
    Ter=tempe_[ir];
    Tel=tempe_[il];
    Tlr=templ_[ir];
    Tll=templ_[il];
    kel=Ke_[symidx(rl, rl)],ker=Ke_[symidx(rr, rr)];
    kll=Kl_[symidx(rl, rl)],klr=Kl_[symidx(rr, rr)];
    ml=(vol == NULL? 1.0f: vol[il]);
    mr=(vol == NULL? 1.0f: vol[ir]);

//      if ((iz==0)||(ml==0)) {   // Left border or no cell at the left (material needs at least two cells!)
      if ((ix==0)) {   // Left border or no cell at the left (material needs at least two cells!)
        lapTex=ke/Ce*(2.0f*Ter-2.0f*Te)/wx/wx;
        lapTlx=kl/Cl*(2.0f*Tlr-2.0f*Tl)/wx/wx;
//      } else if ((iz==Nz-1)||(mr==0)) {   // right border or no cell at the right (material needs at least two cells!)
      } else if ((ix==Nx-1)) {   // right border or no cell at the right (material needs at least two cells!)
        lapTex=ke/Ce*(2.0f*Tel-2.0f*Te)/wx/wx;
        lapTlx=kl/Cl*(2.0f*Tll-2.0f*Tl)/wx/wx;
    } else if (((kel!=ke)||(kll!=kl))&&(ix>0)) { // Non uniform material frontier (left)
        float kappae=kel/ke;
        float kappal=kll/kl;
        float alphae=ke/Ce;
        float alphal=kl/Cl;
        lapTex=alphae/(wx*wx)*(2.0f*kappae/(kappae+1.0f)*Tel-(2.0f+(kappae-1.0f)/(kappae+1.0f))*Te+Ter);
        lapTlx=alphal/(wx*wx)*(2.0f*kappal/(kappal+1.0f)*Tll-(2.0f+(kappal-1.0f)/(kappal+1.0f))*Tl+Tlr);
    } else if (((ker!=ke)||(klr!=kl))&&(ix<Nx-1)) { // Non uniform material frontier (right)
        float kappae=ke/ker;
        float kappal=kl/klr;
        float alphae=ke/Ce;
        float alphal=kl/Cl;
        lapTex=alphae/(wx*wx)*(Tel-(2.0f-(kappae-1.0f)/(kappae+1.0f))*Te+2.0f/(kappae+1.0f)*Ter);
        lapTlx=alphal/(wx*wx)*(Tll-(2.0f-(kappal-1.0f)/(kappal+1.0f))*Tl+2.0f/(kappal+1.0f)*Tlr);
      } else { // Uniform in the center
        lapTex=ke/Ce*(Tel-2.0f*Te+Ter)/(wx*wx);
        lapTlx=kl/Cl*(Tll-2.0f*Tl+Tlr)/(wx*wx);
      }
    
    // y derivatives
    il=idx(ix, iy-1, iz);
    ir=idx(ix, iy+1, iz);
    rl=regions[il];
    rr=regions[ir];
    Ter=tempe_[ir];
    Tel=tempe_[il];
    Tlr=templ_[ir];
    Tll=templ_[il];
    kel=Ke_[symidx(rl, rl)],ker=Ke_[symidx(rr, rr)];
    kll=Kl_[symidx(rl, rl)],klr=Kl_[symidx(rr, rr)];
    ml=(vol == NULL? 1.0f: vol[il]);
    mr=(vol == NULL? 1.0f: vol[ir]);

///      if ((iz==0)||(ml==0)) {   // Left border or no cell at the left (material needs at least two cells!)
      if ((iy==0)) {   // Left border or no cell at the left (material needs at least two cells!)
        lapTey=ke/Ce*(2.0f*Ter-2.0f*Te)/wy/wy;
        lapTly=kl/Cl*(2.0f*Tlr-2.0f*Tl)/wy/wy;
//      } else if ((iz==Nz-1)||(mr==0)) {   // right border or no cell at the right (material needs at least two cells!)
      } else if ((iy==Ny-1)) {   // right border or no cell at the right (material needs at least two cells!)
        lapTey=ke/Ce*(2.0f*Tel-2.0f*Te)/wy/wy;
        lapTly=kl/Cl*(2.0f*Tll-2.0f*Tl)/wy/wy;
    } else if (((kel!=ke)||(kll!=kl))&&(iy>0)) { // Non uniform material frontier (left)
        float kappae=kel/ke;
        float kappal=kll/kl;
        float alphae=ke/Ce;
        float alphal=kl/Cl;
        lapTey=alphae/(wy*wy)*(2.0f*kappae/(kappae+1.0f)*Tel-(2.0f+(kappae-1.0f)/(kappae+1.0f))*Te+Ter);
        lapTly=alphal/(wy*wy)*(2.0f*kappal/(kappal+1.0f)*Tll-(2.0f+(kappal-1.0f)/(kappal+1.0f))*Tl+Tlr);
    } else if (((ker!=ke)||(klr!=kl))&&(iy<Ny-1)) { // Non uniform material frontier (right)
        float kappae=ke/ker;
        float kappal=kl/klr;
        float alphae=ke/Ce;
        float alphal=kl/Cl;
        lapTey=alphae/(wy*wy)*(Tel-(2.0f-(kappae-1.0f)/(kappae+1.0f))*Te+2.0f/(kappae+1.0f)*Ter);
        lapTly=alphal/(wy*wy)*(Tll-(2.0f-(kappal-1.0f)/(kappal+1.0f))*Tl+2.0f/(kappal+1.0f)*Tlr);
      } else { // Uniform in the center
        lapTey=ke/Ce*(Tel-2.0f*Te+Ter)/(wy*wy);
        lapTly=kl/Cl*(Tll-2.0f*Tl+Tlr)/(wy*wy);
      }
    
    // only take vertical derivative for 3D sim
    if (Nz != 1) {
      // z derivatives
      il=idx(ix, iy, iz-1);
      ir=idx(ix, iy, iz+1);
      rl=regions[il];
      rr=regions[ir];
      Ter=tempe_[ir];
      Tel=tempe_[il];
      Tlr=templ_[ir];
      Tll=templ_[il];
      kel=Ke_[symidx(rl, rl)],ker=Ke_[symidx(rr, rr)];
      kll=Kl_[symidx(rl, rl)],klr=Kl_[symidx(rr, rr)];
      ml=(vol == NULL? 1.0f: vol[il]);
      mr=(vol == NULL? 1.0f: vol[ir]);

//      if ((iz==0)||(ml==0)) {   // Left border or no cell at the left (material needs at least two cells!)
      if ((iz==0)) {   // Left border or no cell at the left (material needs at least two cells!)
        lapTez=ke/Ce*(2.0f*Ter-2.0f*Te)/wz/wz;
        lapTlz=kl/Cl*(2.0f*Tlr-2.0f*Tl)/wz/wz;
//      } else if ((iz==Nz-1)||(mr==0)) {   // right border or no cell at the right (material needs at least two cells!)
      } else if ((iz==Nz-1)) {   // right border or no cell at the right (material needs at least two cells!)
        lapTez=ke/Ce*(2.0f*Tel-2.0f*Te)/wz/wz;
        lapTlz=kl/Cl*(2.0f*Tll-2.0f*Tl)/wz/wz;
    } else if (((kel!=ke)||(kll!=kl))&&(iz>0)) { // Non uniform material frontier (left)
        float kappae=kel/ke;
        float kappal=kll/kl;
        float alphae=ke/Ce;
        float alphal=kl/Cl;
        lapTez=alphae/(wz*wz)*(2.0f*kappae/(kappae+1.0f)*Tel-(2.0f+(kappae-1.0f)/(kappae+1.0f))*Te+Ter);
        lapTlz=alphal/(wz*wz)*(2.0f*kappal/(kappal+1.0f)*Tll-(2.0f+(kappal-1.0f)/(kappal+1.0f))*Tl+Tlr);
    } else if (((ker!=ke)||(klr!=kl))&&(iz<Nz-1)) { // Non uniform material frontier (right)
        float kappae=ke/ker;
        float kappal=kl/klr;
        float alphae=ke/Ce;
        float alphal=kl/Cl;
        lapTez=alphae/(wz*wz)*(Tel-(2.0f-(kappae-1.0f)/(kappae+1.0f))*Te+2.0f/(kappae+1.0f)*Ter);
        lapTlz=alphal/(wz*wz)*(Tll-(2.0f-(kappal-1.0f)/(kappal+1.0f))*Tl+2.0f/(kappal+1.0f)*Tlr);
      } else { // Uniform in the center
        lapTez=ke/Ce*(Tel-2.0f*Te+Ter)/(wz*wz);
        lapTlz=kl/Cl*(Tll-2.0f*Tl+Tlr)/(wz*wz);
      }
    
    }

    dt0e_[i] += lapTex+lapTey+lapTez;
    dt0l_[i] += lapTlx+lapTly+lapTlz;

// Exchange between temperatures
    dt0e_[i]+=-Gel*(Te-Tl)/Ce;
    dt0l_[i]+=-Gel*(Tl-Te)/Cl;

//External sources on electron?
    dt0l_[i]+=dot(J,J)*res/Ce;          //Joule Heating

// Circular dichroism
    float alphaD=dot(cd,cd);
    if (alphaD==0){
	    dt0e_[i]+=Qext/Ce;                  //External Heating source in W/m3 without circular dichoism
	} else
	{

  /*
  //Old IMplementations MCD
	float norm1=sqrt(mm);
	float norm2=sqrt(alphaD);
	float pe = -1.0*dot(m0,cd);		// Inversion of sign to lead to the same results as IFE
	float scaleCD=1.0+(pe/norm1/norm2-1.0)/2.0*norm2;
	dt0e_[i]+=Qext*scaleCD;
  */
  // New Implementation MCD
	float norm2=sqrt(alphaD);
  float pe = -1.0*dot(m0,cd);		// Inversion of sign to lead to the same results as IFE
  if (pe>0) pe=1*norm2;
  if (pe<0) pe=-1*norm2;
  float scaleCD=1.0+0.5*pe;
  dt0e_[i]+=Qext*scaleCD/Ce;
	}

// Missing constants
//    dt0e_[i]=dt0e_[i]/Ce;
//    dt0l_[i]=dt0l_[i]/Cl;

    if (scaletausubs>0) {
    if (Tausubsth!=0) {dt0l_[i]=dt0l_[i]-scaletausubs*(templ_[i]-Tsubsth)/Tausubsth; }  // Substrate effect on lattice?
    }
    }


    else{
      tempe_[i]=0;
      templ_[i]=0;
    }

}
