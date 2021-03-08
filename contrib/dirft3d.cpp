#include <iostream>


void dirft3d1(int lib,int nj,FLT* x,FLT *y,FLT *z,int ix, int iy, int iz,
              CPX* c,int iflag,int ms, int mt, int mu, CPX* f)
/* Direct computation of 3D type-1 nonuniform FFT. Interface same as finufft3d1.
c                     nj-1
c     f[k1,k2,k3] =   SUM  c[j] exp(+-i (k1 x[j] + k2 y[j] + k2 z[j]))
c                     j=0
c
c     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
          -mu/2 <= k3 <= (mu-1)/2,
c     The output array is in increasing k1 ordering (fast), then increasing
      k2 ordering (medium), then increasing k3 (fast). If iflag>0 the + sign is
c     used, otherwise the - sign is used, in the exponential.
*  Uses C++ complex type and winding trick.  Barnett 2/1/17
*/
{
  int k1min = -(ms/2), k2min = -(mt/2), k3min = -(mu/2);   // integer divide
  int N = ms*mt*mu;        // total # output modes
  for (int m=0;m<N;++m) f[m] = CPX(0,0);    // it knows f is complex type
  for (int j=0;j<nj;++j) {            // src pts
	if(lib==3){
      CPX a1 = (iflag>0) ? exp(IMA*(FLT)(2*M_PI*x[j*ix])) : exp(-IMA*(FLT)(2*M_PI*x[j*ix]));
      CPX a2 = (iflag>0) ? exp(IMA*(FLT)(2*M_PI*y[j*iy])) : exp(-IMA*(FLT)(2*M_PI*y[j*iy]));
      CPX a3 = (iflag>0) ? exp(IMA*(FLT)(2*M_PI*z[j*iz])) : exp(-IMA*(FLT)(2*M_PI*z[j*iz]));
      CPX sp3 = pow(a3,(FLT)k3min);
      CPX sp2 = pow(a2,(FLT)k2min);
      CPX p1 = pow(a1,(FLT)k1min);  // starting phase for most neg k2 freq  
      CPX cc = c[j];                // no 1/nj norm
      int m=0;      // output pointer
      for (int m1=0;m1<ms;++m1) {
        CPX p2 = sp2;
        for (int m2=0;m2<mt;++m2) {
	      CPX p3 = sp3;
	      for (int m3=0;m3<mu;++m3) {
	        f[m++] += cc * p1 * p2 * p3;
	        p3 *= a3;
	      }
	      p2 *= a2;
        }
        p1 *= a1;
      }
	}else{
      CPX a1 = (iflag>0) ? exp(IMA*x[j]) : exp(-IMA*x[j]);
      CPX a2 = (iflag>0) ? exp(IMA*y[j]) : exp(-IMA*y[j]);
      CPX a3 = (iflag>0) ? exp(IMA*z[j]) : exp(-IMA*z[j]);
      CPX sp1 = pow(a1,(FLT)k1min);  // starting phase for most neg k1 freq
      CPX sp2 = pow(a2,(FLT)k2min);
      CPX p3 = pow(a3,(FLT)k3min);
      CPX cc = c[j];                 // no 1/nj norm
	  CPX p1;
      int m=0;      // output pointer
      for (int m3=0;m3<mu;++m3) {
        CPX p2 = sp2;
        for (int m2=0;m2<mt;++m2) {
	      CPX p1 = sp1;
	      for (int m1=0;m1<ms;++m1) {
	        f[m++] += cc * p1 * p2 * p3;
	        p1 *= a1;
	      }
	      p2 *= a2;
        }
        p3 *= a3;
      }
	}
  }
}

void dirft3d2(int lib, int nj,FLT* x,FLT *y,FLT *z,int ix, int iy, int iz,
              CPX* c,int iflag,int ms, int mt, int mu, CPX* f)
/* Direct computation of 3D type-2 nonuniform FFT. Interface same as finufft3d2

     c[j] =   SUM    f[k1,k2,k3] exp(+-i (k1 x[j] + k2 y[j] + k3 z[j])) 
            k1,k2,k3  
                            for j = 0,...,nj-1
    where sum is over -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2,
                      -mu/2 <= k3 <= (mu-1)/2.

    The input array is in increasing k1 ordering (fast), then increasing
    k2 ordering (medium), then increasing k3 (fast).
    If iflag>0 the + sign is used, otherwise the - sign is used, in the
    exponential.
    Uses C++ complex type and winding trick.  Barnett 2/1/17
*/
{
  int k1min = -(ms/2), k2min = -(mt/2), k3min = -(mu/2);  // integer divide
  for (int j=0;j<nj;++j) {
	if(lib==3){
      CPX a1 = (iflag>0) ? exp(IMA*(FLT)(2*M_PI*x[j*ix])) : exp(-IMA*(FLT)(2*M_PI*x[j*ix]));
      CPX a2 = (iflag>0) ? exp(IMA*(FLT)(2*M_PI*y[j*iy])) : exp(-IMA*(FLT)(2*M_PI*y[j*iy]));
      CPX a3 = (iflag>0) ? exp(IMA*(FLT)(2*M_PI*z[j*iz])) : exp(-IMA*(FLT)(2*M_PI*z[j*iz]));
      CPX sp3 = pow(a3,(FLT)k3min);
      CPX sp2 = pow(a2,(FLT)k2min);
      CPX p1 = pow(a1,(FLT)k1min);  // starting phase for most neg k2 freq  
      CPX cc = CPX(0,0);
      int m=0;      // input pointer
      for (int m1=0;m1<ms;++m1) {
        CPX p2 = sp2;
        for (int m2=0;m2<mt;++m2) {
	      CPX p3 = sp3;
	      for (int m3=0;m3<mu;++m3) {
	        cc += f[m++] * p1 * p2 * p3;
	        p3 *= a3;
	      }
	      p2 *= a2;
        }
        p1 *= a1;
      }
      c[j] = cc;
	}else{
      CPX a1 = (iflag>0) ? exp(IMA*x[j]) : exp(-IMA*x[j]);
      CPX a2 = (iflag>0) ? exp(IMA*y[j]) : exp(-IMA*y[j]);
      CPX a3 = (iflag>0) ? exp(IMA*z[j]) : exp(-IMA*z[j]);
      CPX sp1 = pow(a1,(FLT)k1min);
      CPX sp2 = pow(a2,(FLT)k2min);
      CPX p3 = pow(a3,(FLT)k3min);
      CPX cc = CPX(0,0);
      int m=0;      // input pointer
      for (int m3=0;m3<mu;++m3) {
        CPX p2 = sp2;
        for (int m2=0;m2<mt;++m2) {
	      CPX p1 = sp1;
	      for (int m1=0;m1<ms;++m1) {
	        cc += f[m++] * p1 * p2 * p3;
	        p1 *= a1;
	      }
	      p2 *= a2;
        }
        p3 *= a3;
      }
      c[j] = cc;
    }
  }
}

void dirft3d3(int nj,FLT* x,FLT *y,FLT *z,CPX* c,int iflag,int nk, FLT* s, FLT* t, FLT *u, CPX* f)
/* Direct computation of 3D type-3 nonuniform FFT. Interface same as finufft3d3
c               nj-1
c     f[k]  =   SUM   c[j] exp(+-i (s[k] x[j] + t[k] y[j] + u[k] z[j]))
c               j=0                   
c                    for k = 0, ..., nk-1
c  If iflag>0 the + sign is used, otherwise the - sign is used, in the
c  exponential. Uses C++ complex type. Simple brute force.  Barnett 2/1/17
*/
{
  for (int k=0;k<nk;++k) {
    CPX ss = (iflag>0) ? IMA*s[k] : -IMA*s[k];
    CPX tt = (iflag>0) ? IMA*t[k] : -IMA*t[k];
    CPX uu = (iflag>0) ? IMA*u[k] : -IMA*u[k];
    f[k] = CPX(0,0);
    for (int j=0;j<nj;++j)
      f[k] += c[j] * exp(ss*x[j] + tt*y[j] + uu*z[j]);
  }
}
