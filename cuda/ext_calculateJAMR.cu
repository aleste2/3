#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"


extern "C"

 __global__ void
calculateJAMR(  float* __restrict__  LocalV_,
  float* __restrict__  LocalSigma_,

  float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
  float* __restrict__ JAMRx, float* __restrict__ JAMRy, float* __restrict__ JAMRz,
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
    if (mm!=0)
    {
      wy=wx;
      wy=wy;
      wz=wz;
      float3 JAMR = make_float3(JAMRx[i], JAMRy[i], JAMRz[i]);
      float Sigma = LocalSigma_[i];

      JAMR=0*m0;
      // Neighbours

      int i_;    // neighbor index
      float3 m_; // neighbor mag
      float mm_;

      // left neighbor
      if (ix-1>=0){
        i_  = idx(ix-1, iy, iz);           // clamps or wraps index according to PBC
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        mm_=dot(m_,m_);
        if (mm_!=0)
        {
         JAMR.x+=LocalV_[i]-LocalV_[i_];
        }
      }

      // right neighbor
      if (ix+1<Nx){
        i_  = idx(ix+1, iy, iz);           // clamps or wraps index according to PBC
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        mm_=dot(m_,m_);
        if (mm_!=0)
        {
         JAMR.x+=LocalV_[i_]-LocalV_[i];
        }
      }

      // back neighbor
      if (iy-1>=0){
        i_  = idx(ix, iy-1, iz);           // clamps or wraps index according to PBC
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        mm_=dot(m_,m_);
        if (mm_!=0)
        {
         JAMR.y+=LocalV_[i]-LocalV_[i_];
        }
      }

      // front neighbor
      if (iy+1<Ny){
        i_  = idx(ix, iy+1, iz);           // clamps or wraps index according to PBC
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        mm_=dot(m_,m_);
        if (mm_!=0)
        {
         JAMR.y+=LocalV_[i_]-LocalV_[i];
        }
      }

      //printf("%e %e\n",Vyr,Vyr);

      // only take vertical derivative for 3D sim
      if (Nz != 1) {
          // bottom neighbor
  	      if (iz-1>=0){
          i_  = idx(ix, iy, iz-1);
          m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
          mm_=dot(m_,m_);
          if (mm_!=0)
          {
          JAMR.z+=LocalV_[i]-LocalV_[i_];
          }
          }

          // top neighbor
          if (iz+1<Nz){
          i_  = idx(ix, iy,iz+1);
          m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
          mm_=dot(m_,m_);
          if (mm_!=0)
          {
          JAMR.z+=LocalV_[i_]-LocalV_[i];
          }
          }
      }
      JAMRx[i]=-Sigma*JAMR.x;
      JAMRy[i]=-Sigma*JAMR.y;
      JAMRz[i]=-Sigma*JAMR.z;
  }

}
