#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"

extern "C"

 __global__ void
evolveAMR( int x1, int x2,
  float* __restrict__  LocalV_,
  float* __restrict__  LocalSigma_,
  float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
  float* __restrict__ JAMRx, float* __restrict__ JAMRy, float* __restrict__ JAMRz,
  float* __restrict__  DeltaRho_, float DeltaRho_mul,
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
    float DeltaRho = amul(DeltaRho_, DeltaRho_mul, i);

    float mm=dot(m0,m0);
    if (mm!=0)
    {
      wy=wx;
      wy=wy;
      wz=wz;
      float3 JAMR = make_float3(JAMRx[i], JAMRy[i], JAMRz[i]);
      float V = LocalV_[i];
      float cost;
      float mJAMR=dot(JAMR,JAMR);
      if (mJAMR!=0) {
        cost=dot(JAMR,m0)/sqrt(dot(JAMR,JAMR));
        //printf("%e %e %e %e %e %e\n",cost,dot(JAMR,m0),JAMRx[i], JAMRy[i], JAMRz[i],sqrt(dot(JAMR,JAMR)));
      } else {
        cost=0;
      }
      float Sigma = 1.0f/(1.0f+DeltaRho*cost*cost);
      LocalSigma_[i]=Sigma;

      // Neighbours
      float Vxl=V,Vxr=V,Vyl=V,Vyr=V,Vzl=V,Vzr=V;
      float Sxl=Sigma,Sxr=Sigma,Syl=Sigma,Syr=Sigma,Szl=Sigma,Szr=Sigma;

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
          Vxl = LocalV_[i_];
          Sxl = LocalSigma_[i_];
        }
      }

      // right neighbor
      if (ix+1<Nx){
        i_  = idx(ix+1, iy, iz);           // clamps or wraps index according to PBC
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        mm_=dot(m_,m_);
        if (mm_!=0)
        {
          Vxr = LocalV_[i_];
          Sxr = LocalSigma_[i_];
        }
      }

      // back neighbor
      if (iy-1>=0){
        i_  = idx(ix, iy-1, iz);           // clamps or wraps index according to PBC
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        mm_=dot(m_,m_);
        if (mm_!=0)
        {
          Vyl = LocalV_[i_];
          Syl = LocalSigma_[i_];
        }
      }

      // front neighbor
      if (iy+1<Ny){
        i_  = idx(ix, iy+1, iz);           // clamps or wraps index according to PBC
        m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
        mm_=dot(m_,m_);
        if (mm_!=0)
        {
          Vyr = LocalV_[i_];
          Syr = LocalSigma_[i_];
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
            Vzl = LocalV_[i_];
            Szl = LocalSigma_[i_];
          }
          }

          // top neighbor
          if (iz+1<Nz){
          i_  = idx(ix, iy,iz+1);
          m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
          mm_=dot(m_,m_);
          if (mm_!=0)
          {
            Vzr = LocalV_[i_];
            Szr = LocalSigma_[i_];
          }
          }
      }

    float wx2=wx*wx*1e18;
    float wy2=wy*wy*1e18;
    float wz2=wz*wz*1e18;
    double A=(Vxr-Vxl)*(Sxr-Sxl)/(2.0f*wx2)+(Vyr-Vyl)*(Syr-Syl)/(2.0f*wy2)+(Vzr-Vzl)*(Szr-Szl)/(2.0f*wz2);
    LocalV_[i]=0.5*(wx2*wy2*wz2)/(wx2*wy2+wx2*wz2+wy2*wz2)*(A/Sigma+((Vxl+Vxr)/wx2+(Vyl+Vyr)/wy2+(Vzl+Vzr)/wz2));
    //printf("%e %e %e %e%e %e \n",DeltaRho,cost,A,Sigma,A/Sigma,((Vxl+Vxr)/wx2+(Vyl+Vyr)/wy2+(Vzl+Vzr)/wz2));
    if (ix==x1) LocalV_[i]=1.0;
    if (ix==x2) LocalV_[i]=0.0f;
  }
}
