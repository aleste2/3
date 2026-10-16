#include "float3.h"
#include "math.h"

// Langevin function
inline __device__ float Brillouin(float x,float J)
{
  return ( (2.0f*J+1.0f)/(2.0f*J)/tanh((2.0f*J+1.0f)/(2.0f*J)*x) - 1.0f/(2.0f*J)/tanh(x/(2.0f*J)));
}

// Langevin derivative function
inline __device__ float DBrillouin(float x,float J)
{
 // if (x<6){
  return ( pow(sinh(x/(2.0f*J)),-2.0f) /(4.0f*J*J) - (1.0f+2.0f*J)*(1.0f+2.0f*J)* pow(sinh((1.0f+2.0f*J)/(2.0f*J)*x),-2.0f) /(4.0f*J*J));
 // }else{
 // return(0.000025073);
 // }
}

// Langevin derivative function (de verdad)
inline __device__ float Lder(float x)
{
	return (1.0f/(x*x)-pow(1.0f/sinh(x),2.0f));
}



// Trial single



// Langevin function
inline __device__ float dL(float x)
{

  float temp = exp(2*x)+(float)1.0;
  if(fabs(temp)>1e38) return x>0 ? 1.0 : -1.0; // Large input
  temp /= temp-2; // temp = coth(x);
  //return 0.0001f+(temp-1/x)/1.05;
  float value=temp-(float)1.0/x;
  if (value<0.03) value=0.01;

  return value;
}

// Langevin derivative function (de verdad)
inline __device__ float dLder(float x)
{
	float value=1.0f/(x*x)-pow(1.0f/sinh(x),2.0f);
  	if (value<0.03) value=0.01;
	return (value);
}

// Langevin function
inline __device__ float dL0(float x)
{

  float temp = exp(2*x)+(double)1.0;
  if(fabs(temp)>1e38) return x>0 ? 1.0 : -1.0; // Large input
  temp /= temp-2; // temp = coth(x);
  //return 0.0001f+(temp-1/x)/1.05;
  float value=temp-(float)1.0/x;

  return value;
}

// Langevin derivative function (de verdad)
inline __device__ float dLder0(float x)
{
	float value=1.0f/(x*x)-pow(1.0f/sinh(x),2.0f);
	return (value);
}


// New Langevin and derivative 2022

inline __device__ float NL(float x)
{
 if (x==0) {return(0.0f);} else {return(1.0f/tanh(x)-1/x);}
}

inline __device__ float DNL(float x)
{
  if (fabs(x)<0.08) {
    return(0.33333-x*x*0.068);} else {
      return (-pow(sinh(x),-2.0f) + 1.0f/(x*x) );}
}

// Claude Functions

inline __device__ float meLLB(float T, float Tc) {
    if (T >= Tc) return 0.0f;
    float me = powf(1.0f - powf(T/Tc, 1.23318f), 0.433936f);
    return fmaxf(me, 0.001f);
}

inline __device__ float LangevinInv(float y) {   // B^{-1}(y)
    if (y < 0.05f) { float y2 = y*y; return 3.0f*y*(1.0f + 0.6f*y2 + 0.566f*y2*y2); }
    y = fminf(y, 0.999f);
    float x = y*(3.0f - y*y)/(1.0f - y*y);       // aproximación de Cohen
    for (int k = 0; k < 3; k++) x -= (NL(x) - y)/DNL(x);
    return x;
}

inline __device__ float chiParLLB(float T, float me, float mua) {   // 1/T
    const float kB = 1.38064852e-23f;
    float x = LangevinInv(me);
    float d = DNL(x);
    float den = (me < 0.05f) ? 1.2f*me*me*(1.0f + 0.94f*me*me)   // serie, evita cancelación
                             : 1.0f - d*x/me;
    return (mua/(kB*T)) * d / den;
}

// me en LLB segun claude
inline __device__ void rSerie(float x, float *r, float *dr) {   // B(x)/x = 1/3 - r(x)
    float x2 = x*x;
    *r  = x2*(1.0f/45.0f - x2*(2.0f/945.0f - x2*(1.0f/4725.0f - x2*(2.0f/93555.0f - x2*1382.0f/638512875.0f))));
    *dr = x *(2.0f/45.0f - x2*(8.0f/945.0f - x2*(6.0f/4725.0f - x2*(16.0f/93555.0f - x2*13820.0f/638512875.0f))));
}

inline __device__ float meMF(float T, float Tc) {   // m = B(3 Tc m / T)
    if (T >= Tc) return 0.0f;
    float lam = 3.0f*Tc/T, c0 = (Tc - T)/T;
    float m = fminf(1.0f, 1.3f*powf(1.0f - powf(T/Tc, 1.23318f), 0.433936f)); // solo punto de partida
    for (int k = 0; k < 30; k++) {          // Newton monótono desde la derecha
        float x = lam*m, h, dh;
        if (x < 1.0f) { float r, dr; rSerie(x, &r, &dr); h = c0 - lam*r; dh = -lam*lam*dr; }
        else { float s = NL(x)/x; h = lam*s - 1.0f; dh = lam*lam*(DNL(x) - s)/x; }
        float dm = -h/dh;
        m = fmaxf(m + dm, 1e-4f);
        if (fabsf(dm) < 1e-4f*m) break;
    }
    return m;
}

inline __device__ float chiParMF(float T, float Tc, float me, float mua) {   // 1/T
    const float kB = 1.38064852e-23f;
    float lam = 3.0f*Tc/T, x = lam*me, d = DNL(x), den;
    if (x < 1.0f) { float r, dr; rSerie(x, &r, &dr); den = lam*x*dr; }  // = 1 − λB'(x) en la raíz
    else den = 1.0f - lam*d;
    return (mua/(kB*T))*d/den;
}
