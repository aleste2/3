#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"

extern "C"

 __global__ void
evaldt02T(float* __restrict__  tempe_,      float* __restrict__ dt0e_,
	float* __restrict__  templ_,      float* __restrict__ dt0l_,
	float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
                float* __restrict__ Ke_, float Ke_mul,
                float* __restrict__ Ce_, float Ce_mul,
                float* __restrict__ Kl_, float Kl_mul,
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
            		float wx, float wy, float wz, int Nx, int Ny, int Nz
                ) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    // central cell
    int i = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[i], my[i], mz[i]);

    float mm=dot(m0,m0);
    dt0e_[i]=0.0;
    dt0l_[i]=0.0;

    if (mm!=0)
    {

    float3 J = vmul(jx_, jy_, jz_, jx_mul, jy_mul, jz_mul, i);
    float Ke = amul(Ke_, Ke_mul, i);
    float Ce = amul(Ce_, Ce_mul, i);
    float Kl = amul(Kl_, Kl_mul, i);
    float Cl = amul(Cl_, Cl_mul, i);
    float Dth = amul(Dth_, Dth_mul, i);
    float Gel = amul(Gel_, Gel_mul, i);
    float3 cd = vmul(cdx_, cdy_, cdz_, cdx_mul, cdy_mul, cdz_mul, i);

    float Tsubsth = amul(Tsubsth_, Tsubsth_mul, i);
    float Tausubsth = amul(Tausubsth_, Tausubsth_mul, i);
    float res = amul(res_, res_mul, i);
    float Qext = amul(Qext_, Qext_mul, i);

    float tempe = tempe_[i];
    if (tempe!=0) {Ce=Ce*tempe/300.0;} else {Ce=Ce*1.0/300.0;}  // To account for temperature dependence. Cl is almost constat at T>TD
    float templ = templ_[i];

    int i_;    // neighbor index
    float3 m_; // neighbor mag
    float mm_;

    // Difusission for each temperature

    float tempve=0;
    float tempvl=0;

    // left neighbor
    if (ix-1>=0){
    i_  = idx(ix-1, iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    mm_=dot(m_,m_);

    if (mm_!=0)
    {
     tempve = tempe_[i_];
     tempvl = templ_[i_];
     dt0e_[i] += (Ke*(tempve-tempe)/wx/wx);
     dt0l_[i] += (Kl*(tempvl-templ)/wx/wx);
     //if (tempv>=1e4) {   printf("%d %d %d %e %e %e %e\n",ix,iy,iz,dt0_[i],temp,tempv,mm); dt0_[i]=0.0;}
    }
    }

    // right neighbor
    if (ix+1<Nx){
    i_  = idx(ix+1, iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    mm_=dot(m_,m_);
    if (mm_!=0)
    {
     tempve = tempe_[i_];
     tempvl = templ_[i_];
     dt0e_[i] += (Ke*(tempve-tempe)/wx/wx);
     dt0l_[i] += (Kl*(tempvl-templ)/wx/wx);
    }
    }

    // back neighbor
    if (iy-1>=0){
    i_  = idx(ix, iy-1, iz);          // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    mm_=dot(m_,m_);
    if (mm_!=0)
    {
     tempve = tempe_[i_];
     tempvl = templ_[i_];
     dt0e_[i] += (Ke*(tempve-tempe)/wy/wy);
     dt0l_[i] += (Kl*(tempvl-templ)/wy/wy);
    }
    }

    // front neighbor

    if (iy+1<Ny){
    i_  = idx(ix, iy+1, iz);          // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    mm_=dot(m_,m_);
    if (mm_!=0)
    {
     tempve = tempe_[i_];
     tempvl = templ_[i_];
     dt0e_[i] += (Ke*(tempve-tempe)/wy/wy);
     dt0l_[i] += (Kl*(tempvl-templ)/wy/wy);
    }
    }

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
	if (iz-1>=0){
        i_  = idx(ix, iy, iz-1);
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        mm_=dot(m_,m_);
        if (mm_!=0)
        {
	     tempve = tempe_[i_];
	     tempvl = templ_[i_];
	     dt0e_[i] += (Ke*(tempve-tempe)/wz/wz);
	     dt0l_[i] += (Kl*(tempvl-templ)/wz/wz);
        }
        }

        // top neighbor
        if (iz+1<Nz){
        i_  = idx(ix, iy,iz+1);
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        mm_=dot(m_,m_);
        if (mm_!=0)
        {
	     tempve = tempe_[i_];
	     tempvl = templ_[i_];
	     dt0e_[i] += (Ke*(tempve-tempe)/wz/wz);
	     dt0l_[i] += (Kl*(tempvl-templ)/wz/wz);
        }
        }
    }

// Exchange between temperatures
    dt0e_[i]+=-Gel*(tempe-templ);
    dt0l_[i]+=-Gel*(templ-tempe);

//External sources on electron?
    dt0l_[i]+=dot(J,J)*res;          //Joule Heating
// Circular dichroism
    float alphaD=dot(cd,cd);
    if (alphaD==0){
	    dt0e_[i]+=Qext;                  //External Heating source in W/m3 without circular dichoism
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
  dt0e_[i]+=Qext*scaleCD;
	}

// Missing constants
    dt0e_[i]=dt0e_[i]/Ce;
    dt0l_[i]=dt0l_[i]/Cl;

    if (Tausubsth!=0) {dt0l_[i]=dt0l_[i]-(templ-Tsubsth)/Tausubsth; }  // Substrate effect on lattice?

    }

}
