#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"

extern "C"

 __global__ void
evalmusStep(float* __restrict__ dmusx, float* __restrict__ dmusy, float* __restrict__ dmusz,
	float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
    float* __restrict__ musx, float* __restrict__ musy, float* __restrict__ musz,
                float* __restrict__ rhosd_, float rhosd_mul,
                float* __restrict__ tausf_, float tausf_mul,
                float* __restrict__ dbar_, float dbar_mul,
                float* __restrict__ sigmabar_, float sigmabar_mul,
                float* __restrict__ dy11x, float* __restrict__ dy11y, float* __restrict__ dy11z,
                float* __restrict__ dy1x, float* __restrict__ dy1y, float* __restrict__ dy1z,
                float dt,
           		float wx, float wy, float wz, int Nx, int Ny, int Nz,
                float* __restrict__ vol,
                uint8_t* __restrict__ regions
                ) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    // central cell
    int i = idx(ix, iy, iz);
	float mm = (vol == NULL? 1.0f: vol[i]);  // To check
    if (mm==0) {
        dmusx[i]=0.0f;
        dmusy[i]=0.0f;
        dmusz[i]=0.0f;
        return;
    }

    // Not working holes!!!!!  mmm=1 mmp=1
    float e=1.6e-19;
    float lapmux,lapmuy,lapmuz,mmm,mmp;
    int im,ip;
    // Lap_x
    float lapmuxx,lapmuxy,lapmuxz;
    im = idx(ix-1, iy, iz);
    ip = idx(ix+1, iy, iz);
    mmm=0;
    mmp=0;
    if (ix>0) {mmm=vol[im];}//mmm=sqrt(mx[im]*mx[im]+my[im]*my[im]+mz[im]*mz[im]);} 
    if (ix<Nx-1) {mmp=vol[ip];}//mmp=sqrt(mx[ip]*mx[ip]+my[ip]*my[ip]+mz[ip]*mz[ip]);} 
    mmm=1;
    mmp=1;
    if ((ix == 0)||(mmm==0)) { // No flux at x=0 or no neighbout at the left
        lapmuxx = (2.0f*musx[ip] - 2.0f * musx[i]) / (wx * wx);
        lapmuxy = (2.0f*musy[ip] - 2.0f * musy[i]) / (wx * wx);
        lapmuxz = (2.0f*musz[ip] - 2.0f * musz[i]) / (wx * wx);
    };
    if ((ix == Nx - 1)||(mmp==0)) { // No flux at the end or no neigbour at the right
        lapmuxx = (2.0f*musx[im] - 2.0f * musx[i]) / (wx * wx);
        lapmuxy = (2.0f*musy[im] - 2.0f * musy[i]) / (wx * wx);
        lapmuxz = (2.0f*musz[im] - 2.0f * musz[i]) / (wx * wx);
    }
    if ((ix > 0) && (ix < Nx - 1)) {
        lapmuxx = (musx[im] - 2.0f * musx[i] + musx[ip]) / (wx * wx);
        lapmuxy = (musy[im] - 2.0f * musy[i] + musy[ip]) / (wx * wx);
        lapmuxz = (musz[im] - 2.0f * musz[i] + musz[ip]) / (wx * wx);
        // Non uniform material
        float sigma = amul(sigmabar_, sigmabar_mul, i);
        float sigmap = amul(sigmabar_, sigmabar_mul, ip);
        float sigmam = amul(sigmabar_, sigmabar_mul, im);
        if (sigma!=sigmap) {
            float jdiffm=2.0f*sigma/e;
            float jdiffp=2.0f*sigmap/e;
            float jbarm=jdiffm/(jdiffm+jdiffp);
            float jbarp=jdiffp/(jdiffm+jdiffp);
            float migmx=-(-musx[i]*jbarm+jbarp*musx[i]-2.0f*jbarp*musx[ip]);
            float migmy=-(-musy[i]*jbarm+jbarp*musy[i]-2.0f*jbarp*musy[ip]);
            float migmz=-(-musz[i]*jbarm+jbarp*musz[i]-2.0f*jbarp*musz[ip]);
            lapmuxx=(musx[im] - 2.0f * musx[i] + migmx) / (wx * wx);
            lapmuxy=(musy[im] - 2.0f * musy[i] + migmy) / (wx * wx);
            lapmuxz=(musz[im] - 2.0f * musz[i] + migmz) / (wx * wx);
        }
        if (sigmam!=sigma) {
            float jdiffm=2.0f*sigmam/e;
            float jdiffp=2.0f*sigma/e;
            float jbarm=jdiffm/(jdiffm+jdiffp);
            float jbarp=jdiffp/(jdiffm+jdiffp);
            float migpx=-(-2.0f*musx[im]*jbarm+jbarm*musx[i]-jbarp*musx[i]);
            float migpy=-(-2.0f*musy[im]*jbarm+jbarm*musy[i]-jbarp*musy[i]);
            float migpz=-(-2.0f*musz[im]*jbarm+jbarm*musz[i]-jbarp*musz[i]);
            lapmuxx=(migpx - 2.0f * musx[i] + musx[ip]) / (wx * wx);
            lapmuxz=(migpy - 2.0f * musy[i] + musy[ip]) / (wx * wx);
            lapmuxz=(migpz - 2.0f * musz[i] + musz[ip]) / (wx * wx);
        }
    }

    // Lap_y
    float lapmuyx,lapmuyy,lapmuyz;
    im = idx(ix, iy-1, iz);
    ip = idx(ix, iy+1, iz);
    mmm=0;
    mmp=0;
    if (iy>0) {mmm=vol[im];}//{mmm=sqrt(mx[im]*mx[im]+my[im]*my[im]+mz[im]*mz[im]);} 
    if (iy<Ny-1) {mmp=vol[ip];}//{mmp=sqrt(mx[ip]*mx[ip]+my[ip]*my[ip]+mz[ip]*mz[ip]);} 
    mmm=1;
    mmp=1;
    if ((iy == 0)||(mmm==0)) { // No flux at x=0 or no neighbout at the left
        lapmuyx = (2.0f*musx[ip] - 2.0f * musx[i]) / (wy * wy);
        lapmuyy = (2.0f*musy[ip] - 2.0f * musy[i]) / (wy * wy);
        lapmuyz = (2.0f*musz[ip] - 2.0f * musz[i]) / (wy * wy);
    };
    if ((iy == Ny - 1)||(mmp==0)) { // No flux at the end or no neigbour at the right
        lapmuyx = (2.0f*musx[im] - 2.0f * musx[i]) / (wy * wy);
        lapmuyy = (2.0f*musy[im] - 2.0f * musy[i]) / (wy * wy);
        lapmuyz = (2.0f*musz[im] - 2.0f * musz[i]) / (wy * wy);
    }
    if ((iy > 0) && (iy < Ny - 1)) {
        lapmuyx = (musx[im] - 2.0f * musx[i] + musx[ip]) / (wy * wy);
        lapmuyy = (musy[im] - 2.0f * musy[i] + musy[ip]) / (wy * wy);
        lapmuyz = (musz[im] - 2.0f * musz[i] + musz[ip]) / (wy * wy);
        // Non uniform material
        float sigma = amul(sigmabar_, sigmabar_mul, i);
        float sigmap = amul(sigmabar_, sigmabar_mul, ip);
        float sigmam = amul(sigmabar_, sigmabar_mul, im);
        if (sigma!=sigmap) {
            float jdiffm=2.0f*sigma/e;
            float jdiffp=2.0f*sigmap/e;
            float jbarm=jdiffm/(jdiffm+jdiffp);
            float jbarp=jdiffp/(jdiffm+jdiffp);
            float migmx=-(-musx[i]*jbarm+jbarp*musx[i]-2.0f*jbarp*musx[ip]);
            float migmy=-(-musy[i]*jbarm+jbarp*musy[i]-2.0f*jbarp*musy[ip]);
            float migmz=-(-musz[i]*jbarm+jbarp*musz[i]-2.0f*jbarp*musz[ip]);
            lapmuyx=(musx[im] - 2.0f * musx[i] + migmx) / (wy * wy);
            lapmuyy=(musy[im] - 2.0f * musy[i] + migmy) / (wy * wy);
            lapmuyz=(musz[im] - 2.0f * musz[i] + migmz) / (wy * wy);
        }
        if (sigmam!=sigma) {
            float jdiffm=2.0f*sigmam/e;
            float jdiffp=2.0f*sigma/e;
            float jbarm=jdiffm/(jdiffm+jdiffp);
            float jbarp=jdiffp/(jdiffm+jdiffp);
            float migpx=-(-2.0f*musx[im]*jbarm+jbarm*musx[i]-jbarp*musx[i]);
            float migpy=-(-2.0f*musy[im]*jbarm+jbarm*musy[i]-jbarp*musy[i]);
            float migpz=-(-2.0f*musz[im]*jbarm+jbarm*musz[i]-jbarp*musz[i]);
            lapmuyx=(migpx - 2.0f * musx[i] + musx[ip]) / (wy * wy);
            lapmuyy=(migpy - 2.0f * musy[i] + musy[ip]) / (wy * wy);
            lapmuyz=(migpz - 2.0f * musz[i] + musz[ip]) / (wy * wy);
        }
    }

    // Lap_z
    float lapmuzx,lapmuzy,lapmuzz;
    im = idx(ix, iy, iz-1);
    ip = idx(ix, iy, iz+1);
    mmm=0;
    mmp=0;
    if (iz>0) {mmm=vol[im];}//{mmm=sqrt(mx[im]*mx[im]+my[im]*my[im]+mz[im]*mz[im]);} 
    if (iz<Nz-1) {mmp=vol[ip];}//{mmp=sqrt(mx[ip]*mx[ip]+my[ip]*my[ip]+mz[ip]*mz[ip]);} 
    mmm=1;
    mmp=1;
    if ((iz == 0)||(mmm==0)) { // No flux at x=0 or no neighbout at the left
        lapmuzx = (2.0f*musx[ip] - 2.0f * musx[i]) / (wz * wz);
        lapmuzy = (2.0f*musy[ip] - 2.0f * musy[i]) / (wz * wz);
        lapmuzz = (2.0f*musz[ip] - 2.0f * musz[i]) / (wz * wz);
    };
    if ((iz == Nz - 1)||(mmp==0)) { // No flux at the end or no neigbour at the right
        lapmuzx = (2.0f*musx[im] - 2.0f * musx[i]) / (wz * wz);
        lapmuzy = (2.0f*musy[im] - 2.0f * musy[i]) / (wz * wz);
        lapmuzz = (2.0f*musz[im] - 2.0f * musz[i]) / (wz * wz);
    }
    if ((iz > 0) && (iz < Nz - 1)) {
        lapmuzx = (musx[im] - 2.0f * musx[i] + musx[ip]) / (wz * wz);
        lapmuzy = (musy[im] - 2.0f * musy[i] + musy[ip]) / (wz * wz);
        lapmuzz = (musz[im] - 2.0f * musz[i] + musz[ip]) / (wz * wz);
        // Non uniform material
        float sigma = amul(sigmabar_, sigmabar_mul, i);
        float sigmap = amul(sigmabar_, sigmabar_mul, ip);
        float sigmam = amul(sigmabar_, sigmabar_mul, im);
        if (sigma!=sigmap) {
            float jdiffm=2.0f*sigma/e;
            float jdiffp=2.0f*sigmap/e;
            float jbarm=jdiffm/(jdiffm+jdiffp);
            float jbarp=jdiffp/(jdiffm+jdiffp);
            float migmx=-(-musx[i]*jbarm+jbarp*musx[i]-2.0f*jbarp*musx[ip]);
            float migmy=-(-musy[i]*jbarm+jbarp*musy[i]-2.0f*jbarp*musy[ip]);
            float migmz=-(-musz[i]*jbarm+jbarp*musz[i]-2.0f*jbarp*musz[ip]);
            lapmuzx=(musx[im] - 2.0f * musx[i] + migmx) / (wz * wz);
            lapmuzy=(musy[im] - 2.0f * musy[i] + migmy) / (wz * wz);
            lapmuzz=(musz[im] - 2.0f * musz[i] + migmz) / (wz * wz);
        }
        if (sigmam!=sigma) {
            float jdiffm=2.0f*sigmam/e;
            float jdiffp=2.0f*sigma/e;
            float jbarm=jdiffm/(jdiffm+jdiffp);
            float jbarp=jdiffp/(jdiffm+jdiffp);
            float migpx=-(-2.0f*musx[im]*jbarm+jbarm*musx[i]-jbarp*musx[i]);
            float migpy=-(-2.0f*musy[im]*jbarm+jbarm*musy[i]-jbarp*musy[i]);
            float migpz=-(-2.0f*musz[im]*jbarm+jbarm*musz[i]-jbarp*musz[i]);
            lapmuzx=(migpx - 2.0f * musx[i] + musx[ip]) / (wz * wz);
            lapmuzy=(migpy - 2.0f * musy[i] + musy[ip]) / (wz * wz);
            lapmuzz=(migpz - 2.0f * musz[i] + musz[ip]) / (wz * wz);
        }
    }

    lapmux=lapmuxx+lapmuyx+lapmuzx;
    lapmuy=lapmuxy+lapmuyy+lapmuzy;
    lapmuz=lapmuxz+lapmuyz+lapmuzz;

    float GammaLL= 1.7595e11;
    float Dt_si=dt/GammaLL;
    float moldx=mx[i]-dt*(dy1x[i]); // LL constant!
    float moldy=my[i]-dt*(dy1y[i]); // LL constant!
    float moldz=mz[i]-dt*(dy1z[i]); // LL constant!
    float mnewx=mx[i]+dt*(dy11x[i]*0.5f-0.5f*dy1x[i]); // LL constant!
    float mnewy=my[i]+dt*(dy11y[i]*0.5f-0.5f*dy1y[i]); // LL constant!
    float mnewz=mz[i]+dt*(dy11z[i]*0.5f-0.5f*dy1z[i]); // LL constant!
    float mmnew=sqrt(mnewx*mnewx+mnewy*mnewy+mnewz*mnewz);
    float mmold=sqrt(moldx*moldx+moldy*moldy+moldz*moldz);
    float dmdt=(mmnew-mmold)/Dt_si;
    float tausf = amul(tausf_, tausf_mul, i);
    float rhosd = amul(rhosd_, rhosd_mul, i);
    float Ddiff = amul(dbar_, dbar_mul, i);

    float ux=0,uy=0,uz=0,norm=sqrt(mx[i]*mx[i]+my[i]*my[i]+mz[i]*mz[i]);
    if (norm>0) {
        ux=mx[i]/norm;
        uy=my[i]/norm;
        uz=mz[i]/norm;
    }
    //if (i==8) printf("mu_s:%e Lap: %e deltam: %e mnew:%e mold:%e\n", musz[i], lapmu,deltam,mmnew,mmold);
    //dmusz[i]=GammaLL*rhosd*dy1z[i] - musz[i] / tausf + Ddiff*(lapmu);
    dmusx[i]=ux*rhosd*dmdt - musx[i] / tausf + Ddiff*(lapmux);
    dmusy[i]=uy*rhosd*dmdt - musy[i] / tausf + Ddiff*(lapmuy);
    dmusz[i]=uz*rhosd*dmdt - musz[i] / tausf + Ddiff*(lapmuz);

    //dmusx[i]=GammaLL*rhosd*dy1x[i] - musx[i] / tausf + Ddiff*(lapmux);
    //dmusy[i]=GammaLL*rhosd*dy1y[i] - musy[i] / tausf + Ddiff*(lapmuy);
}
