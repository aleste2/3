#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "math.h"

// See exchange.go for more details.
extern "C" __global__ void
addexchangeato(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float* __restrict__ Mu_, float Mu_mul,
            float* __restrict__ Nv_, float Nv_mul,
            float* __restrict__ aLUT2d,float* __restrict__ dmiLUT2d, uint8_t* __restrict__ regions,
            int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    // central cell
    int I = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[I], my[I], mz[I]);

    if (is0(m0)) {
        return;
    }

    uint8_t r0 = regions[I];
    float3 B  = make_float3(0.0,0.0,0.0);
    float  Nv  = amul(Nv_, Nv_mul, I);

    int i_;    // neighbor index
    float3 m_; // neighbor mag
    float a__; // inter-cell exchange stiffness
    float dmi__; // inter-cell exchange stiffness
    //float dx;
    //float dy;
    float modulo;
    float3 ud=make_float3(0.0,0.0,1.0); // DMI direction
    float3 uij=make_float3(0.0,0.0,1.0); // uij
    float3 dxuij;
    float3 Bdmi;

    // general cell 27 neighbours
    for (int ii=-1;ii<2;ii++) {
        for (int jj=-1;jj<2;jj++) {
            for (int kk=-1;kk<2;kk++) {
                if ((ii!=0)||(jj!=0)||(kk!=0))
                    if ((Nv!=6)||(ii*ii+jj*jj+kk*kk==1)){   // For 6 neighbours only add only of changes (x xor y xor z)
	                   int ie,je,ke;
	                   ie=ix;
	                   if ( ii==-1) {ie=lclampx(ix-1);}
	                   if ( ii==1)  {ie=hclampx(ix+1);}
	                   je=iy;
	                   if ( jj==-1) {je=lclampy(iy-1);}
	                   if ( jj==1)  {je=hclampy(iy+1);}
	                   ke=iz;
	                   if ( kk==-1) {ke=lclampz(iz-1);}
	                   if ( kk==1)  {ke=hclampz(iz+1);}
                       i_  = idx(ie, je, ke);           // clamps or wraps index according to PBC
	                   if (regions[i_]!=255) {
                       m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
	                   //m_  = ( is0(m_)? m0: m_ );                  // do not replace missing non-boundary neighbor and cancel m out of volume of not PBC
                       if (!PBCx) {
                            if (((ix+ii)<0)||((ix+ii)>Nx-1)) {
                                m_  = make_float3(0.0, 0.0, 0.0);
                            }
                        }
                        if (!PBCy) {
                            if (((iy+jj)<0)||((iy+jj)>Ny-1)) {
                                m_  = make_float3(0.0, 0.0, 0.0);
                            }
                        }
                        if (!PBCz) {
                            if (((iz+kk)<0)||((iz+kk)>Nz-1)) {
                                m_  = make_float3(0.0, 0.0, 0.0);
                            }
                        }
                        a__ = aLUT2d[symidx(r0, regions[i_])];
                        dmi__ = dmiLUT2d[symidx(r0, regions[i_])];
                        if (dmi__!=0){
                            modulo=sqrt(1.0*ii*ii+jj*jj+kk*kk);
                            uij=1.0f/modulo*make_float3(ii,jj,kk); 
                            dxuij=cross(ud,uij);
                            Bdmi=dmi__*cross(dxuij,m_);
                            B+=Bdmi;
 //Works as Mumax Micromag
 /*           modulo=sqrt(1.0*ii*ii+jj*jj+kk*kk);
            dx=ii/modulo;
            dy=jj/modulo;
            B.x+=dmi__*(m_.z*dx);
            B.y+=dmi__*(m_.z*dy);
            B.z+=dmi__*(-m_.x*dx-m_.y*dy);
*/
                        }
                        B += a__ *m_;  /// change
                    }
                }
            }
        }
    }

/*
// Six neighbours
if (Nv==6) {
    int ii,jj,kk;
    // left neighbor
    i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
    //m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
        ii=-1;
        jj=0;
        kk=0;
    if (!PBCx) {
        if (((ix+ii)<0)||((ix+ii)>Nx-1)) {
            m_  = make_float3(0.0, 0.0, 0.0);
        }
    }
    a__ = aLUT2d[symidx(r0, regions[i_])];
	B += a__ *m_;
    dmi__ = dmiLUT2d[symidx(r0, regions[i_])];
    if (dmi__!=0){

        modulo=sqrt(1.0*ii*ii+jj*jj+kk*kk);
        uij=1.0f/modulo*make_float3(ii,jj,kk); 
        dxuij=cross(ud,uij);
        Bdmi=dmi__*cross(dxuij,m_);
        B+=Bdmi;
    }


    // right neighbor
    i_  = idx(hclampx(ix+1), iy, iz);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    //m_  = ( is0(m_)? m0: m_ );
        ii=1;
        jj=0;
        kk=0;
    if (!PBCx) {
        if (((ix+ii)<0)||((ix+ii)>Nx-1)) {
            m_  = make_float3(0.0, 0.0, 0.0);
        }
    }
    a__ = aLUT2d[symidx(r0, regions[i_])];
	B += a__ *m_;
    dmi__ = dmiLUT2d[symidx(r0, regions[i_])];
    if (dmi__!=0){
        modulo=sqrt(1.0*ii*ii+jj*jj+kk*kk);
        uij=1.0f/modulo*make_float3(ii,jj,kk); 
        dxuij=cross(ud,uij);
        Bdmi=dmi__*cross(dxuij,m_);
        B+=Bdmi;
    }

    // back neighbor
    i_  = idx(ix, lclampy(iy-1), iz);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    //m_  = ( is0(m_)? m0: m_ );
        ii=0;
        jj=-1;
        kk=0;
               if (!PBCy) {
        if (((iy+jj)<0)||((iy+jj)>Ny-1)) {
            m_  = make_float3(0.0, 0.0, 0.0);
        }
       }
    a__ = aLUT2d[symidx(r0, regions[i_])];
 	B += a__ *m_;
    dmi__ = dmiLUT2d[symidx(r0, regions[i_])];
    if (dmi__!=0){
        modulo=sqrt(1.0*ii*ii+jj*jj+kk*kk);
        uij=1.0f/modulo*make_float3(ii,jj,kk); 
        dxuij=cross(ud,uij);
        Bdmi=dmi__*cross(dxuij,m_);
        B+=Bdmi;
    }

    // front neighbor
    i_  = idx(ix, hclampy(iy+1), iz);
    m_  = make_float3(mx[i_], my[i_], mz[i_]);
    //m_  = ( is0(m_)? m0: m_ );
        ii=0;
        jj=-1;
        kk=0;
       if (!PBCy) {
        if (((iy+jj)<0)||((iy+jj)>Ny-1)) {
            m_  = make_float3(0.0, 0.0, 0.0);
        }
       }
    a__ = aLUT2d[symidx(r0, regions[i_])];
	B += a__ *m_;
    dmi__ = dmiLUT2d[symidx(r0, regions[i_])];
    if (dmi__!=0){
        ii=0;
        jj=1;
        kk=0;
        modulo=sqrt(1.0*ii*ii+jj*jj+kk*kk);
        uij=1.0f/modulo*make_float3(ii,jj,kk); 
        dxuij=cross(ud,uij);
        Bdmi=dmi__*cross(dxuij,m_);
        B+=Bdmi;
    }


    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        i_  = idx(ix, iy, lclampz(iz-1));
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        //m_  = ( is0(m_)? m0: m_ );
        ii=0;
        jj=0;
        kk=-1;
       if (!PBCz) {
        if (((iz+kk)<0)||((iz+kk)>Nz-1)) {
            m_  = make_float3(0.0, 0.0, 0.0);
        }
       }
        a__ = aLUT2d[symidx(r0, regions[i_])];
	B += a__ *m_;
    dmi__ = dmiLUT2d[symidx(r0, regions[i_])];
    if (dmi__!=0){

        modulo=sqrt(1.0*ii*ii+jj*jj+kk*kk);
        uij=1.0f/modulo*make_float3(ii,jj,kk); 
        dxuij=cross(ud,uij);
        Bdmi=dmi__*cross(dxuij,m_);
        B+=Bdmi;
    }


        // top neighbor
        i_  = idx(ix, iy, hclampz(iz+1));
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        //m_  = ( is0(m_)? m0: m_ );
        ii=0;
        jj=0;
        kk=1;
       if (!PBCz) {
        if (((iz+kk)<0)||((iz+kk)>Nz-1)) {
            m_  = make_float3(0.0, 0.0, 0.0);
        }
       }
        a__ = aLUT2d[symidx(r0, regions[i_])];
	B += a__ *m_;
    dmi__ = dmiLUT2d[symidx(r0, regions[i_])];
    if (dmi__!=0){

        modulo=sqrt(1.0*ii*ii+jj*jj+kk*kk);
        uij=1.0f/modulo*make_float3(ii,jj,kk); 
        dxuij=cross(ud,uij);
        Bdmi=dmi__*cross(dxuij,m_);
        B+=Bdmi;
    }

    }
}


*/
    float invMu = inv_Msat(Mu_, Mu_mul, I);
    Bx[I] += B.x*invMu;
    By[I] += B.y*invMu;
    Bz[I] += B.z*invMu;
}

