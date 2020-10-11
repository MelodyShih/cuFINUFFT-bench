#include <cuComplex.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>       // std::vector
#include <ctime>        // std::time
#include <cstdlib> 

#ifdef SINGLE

#ifndef FLT
typedef float FLT;
#endif
#ifndef IMA 
#define IMA std::complex<FLT>(0.0,1.0) 
#endif

#ifndef CPX
typedef std::complex<float> CPX;
#endif

#else

#ifndef FLT
typedef double FLT;
#endif
#ifndef IMA 
#define IMA std::complex<double>(0.0,1.0) 
#endif

#ifndef CPX
typedef std::complex<double> CPX;
#endif
#endif

#ifndef rand01
#define rand01() ((FLT)rand()/RAND_MAX)
#endif
#ifndef randm11
#define randm11() (2*rand01() - (FLT)1.0)
#endif
#ifndef crandm11
#define crandm11() (randm11() + IMA*randm11())
#endif

using namespace Eigen;
/* 
 * Create data for Type 1 transformation (nonuniform -> uniform) 
 *
 * 	c[i]   : strength of ith nonuniform pt
 * 	x[i*ix]: x coordinate of ith nonuniform pt
 * 	y[i*iy]: y coordinate of ith nonuniform pt
 * 	z[i*iz]: z coordinate of ith nonuniform pt
 * 	ix, iy, iz: the gap between ith and (i+1)th nonuniform pt data
 *
 */
void gauss(int N, float *x){
	MatrixXd T(N,N);
	T.setZero();
	VectorXd beta(N-1);
	for(int i=0; i<N-1; i++){
		beta(i) = 0.5/(sqrt(1.0 - 1.0/((2*(i+1))*(2*(i+1)))));
	}
	//T.diagonal(1) = beta;
	for(int i=0; i<N; i++){
		if(i<N-1){
			T(i,i+1) = beta(i);
		}
		if(i>0){
			T(i,i-1) = beta(i-1);
		}
	}
	SelfAdjointEigenSolver<MatrixXd> es;
	VectorXd eigvals;
	es.compute(T, /* computeEigenvectors = */ false);
	eigvals = es.eigenvalues();
	//x = eigvals.data();
	for(int i=0; i<N; i++){
		x[i] = eigvals(i);
	}
}

void create_nupts(int nupts_distr, int dim, int M, FLT*x, FLT*y, FLT*z, int ix, 
		int iy, int iz, FLT scale, int N1, int N2, int N3)
{
	switch(nupts_distr){
		case 1:
			{
				for (int i = 0; i < M; i++) {
					x[i*ix] = scale*randm11();
					if(dim > 1){
						y[i*iy] = scale*randm11();
					}
					if(dim > 2)
						z[i*iz] = scale*randm11();
				}
			}
			break;
		case 2:
			{
				std::srand ( unsigned ( std::time(0) ) );
  				std::vector<int> vecshuff;

  				// set some values:
  				for (int i=0; i<M; ++i) vecshuff.push_back(i); // 1 2 3 4 5 6 7 8 9
  				// using built-in random generator:
  				//std::random_shuffle ( vecshuff.begin(), vecshuff.end() );
				
				int m = ceil(pow(M, 1.0/dim));
				if(m%2==1) m=m+1;
				float* mug = (float*) malloc(m*sizeof(float));

				for(int i=0; i<m; i++){
					mug[i] = cos(M_PI*((i+1)- 0.5)/m);
				}
				 
				if(dim == 1){
					int n_set = 0;
					for(int i=0; i<m; i++){
						if(i >= M){
							printf("warning: the nupts are trucated (%d)\n", m*m-n_set);
							return;
						}
						x[vecshuff[i]*ix] = scale*mug[i]; 
						n_set++;
					}
				}

				if(dim == 2){
					float* pg  = (float*) malloc(m*sizeof(float));
					for(int i=0; i<m; i++){
						pg[i]  = 2*M_PI*(i+1)/m;
					}
					int n_set = 0;
					for(int i=0; i<m; i++){
						for(int j=0; j<m; j++){
							int idx = i*m+j;
							if(idx >= M){
								printf("warning: the nupts are trucated (%d)\n", m*m-n_set);
								return;
							}
							x[vecshuff[idx]*ix] = scale*mug[j]*cos(pg[i]);
							y[vecshuff[idx]*iy] = scale*mug[j]*sin(pg[i]);
						}
					}
				}
				if(dim == 3){
					int mp = 2*m;
					float* pg  = (float*) malloc(mp*sizeof(float));
					for(int i=0; i<mp; i++){
						pg[i]  = 2*M_PI*(i+1)/mp;
					}
					int mr = m/2; 
					float* rg = (float*) malloc(mr*sizeof(float));
					gauss(mr, rg);
					for(int i=0; i<mr; i++){
						rg[i] = scale*(1+rg[i])/2.0;
					}
					float* sthg = (float*) malloc(m*sizeof(float));
					for(int i=0; i<m; i++){
						sthg[i] = sqrt(1 - mug[i]*mug[i]);
					}
					int n_set = 0;
					for(int r=0; r<mr; r++){
						for(int i=0; i<mp; i++){
							for(int j=0; j<m; j++){
								int idx = r*mp*m + i*m + j;
								if(idx >= M){
									printf("warning: the nupts are trucated (%d)\n", m*m*m-n_set);
									return;
								}
								x[vecshuff[idx]*ix] = rg[r]*cos(pg[i])*sthg[j];
								y[vecshuff[idx]*iy] = rg[r]*sin(pg[i])*sthg[j];
								z[vecshuff[idx]*iz] = rg[r]*mug[j];
								n_set++;
							}
						}
					}
					
				}
			}
			break;
		case 3:
			{
				for (int i = 0; i < M; i++) {
					x[i] = scale*rand01()/N1*8;// x in [-pi,pi)
					if(dim > 1)
						y[i] = scale*rand01()/N2*8;
					if(dim > 2)
						z[i] = scale*rand01()/N3*8;
				}
			}
			break;

		default:
			fprintf(stderr,"Invalid distribution of nonuniform points\n");
	}	
}
void create_data_type1(int nupts_distr, int dim, int M, FLT* x, FLT* y, FLT* z, 
		int ix, int iy, int iz, CPX* c, FLT scale, int N1, int N2, int N3)
{
	create_nupts(nupts_distr, dim, M, x, y, z, ix, iy, iz, scale, N1, N2, N3);
	for (int i = 0; i < M; i++) {
		c[i] = std::complex<float>(1.0,1.0);
	}
}

void create_data_type2(int nupts_distr, int dim, int M, FLT* x, FLT* y, FLT* z, 
		int ix, int iy, int iz, int* Nmodes, CPX* f, FLT scale, int N1, 
        int N2, int N3)
{
	create_nupts(nupts_distr, dim, M, x, y, z, ix, iy, iz, scale, N1, N2, N3);
	for (int i = 0; i < Nmodes[0]*Nmodes[1]*Nmodes[2]; i++) {
		f[i] = std::complex<float>(1.0,1.0);
	}
}

void print_solution_type1(int N1, int N2, int N3, CPX* f)
{
	for(int k=0; k<N3; k++){
		for(int j=0; j<N2; j++){
			for(int i=0; i<N1; i++){
				int idx = i+j*N1+k*N1*N2;
				printf("(%3d, %3d, %3d, %10.4f + %10.4fi)\n", i, j, k, f[idx].real(), f[idx].imag());
			}
		}
	}	
}

FLT infnorm(int n, CPX* a)
// ||a||_infty
{
  FLT nrm = 0.0;
  for (int m=0; m<n; ++m) {
    FLT aa = real(conj(a[m])*a[m]);
    if (aa>nrm) nrm = aa;
  }
  return sqrt(nrm);
}

void accuracy_check_type1(int dim, int iflag, int N1, int N2, int N3, int M, FLT* x, 
	FLT* y, FLT *z, int ix, int iy, int iz, CPX* c, CPX* fk, FLT scale)
{
	int N=N1*N2*N3;
	int nt1 = (int)(0.37*N1), nt2 = (int)(0.26*N2), nt3 = (int) (0.18*N3);  
	// choose some mode index to check
	CPX Ft = CPX(0,0), J = IMA*(FLT)iflag*scale;
	for (int j=0; j<M; ++j){
		if (dim == 2)
			Ft += c[j] * exp(J*(nt1*x[j*ix]+nt2*y[j*iy]));   // crude direct
		if (dim == 3)
			Ft += c[j] * exp(J*(nt1*x[j*ix]+nt2*y[j*iy]+nt3*z[j*iz]));   // crude direct
	}
	int it = N1/2+nt1 + N1*(N2/2+nt2) + N1*N2*(N3/2+nt3);   // index in complex F as 1d array
	printf("[gpu   ] one mode: rel err in F[%ld,%ld] is %.3g\n",(int)nt1,(int)nt2,abs(Ft-fk[it])/infnorm(N,fk));
	printf("[gpu   ] one mode: abs err in F[%ld,%ld] is %.3g\n",(int)nt1,(int)nt2,abs(Ft-fk[it]));
}

void accuracy_check_type2(int dim, int iflag, int N1, int N2, int N3, int M, FLT* x, 
	FLT* y, FLT *z, int ix, int iy, int iz, CPX* c, CPX* fk, FLT scale)
{
	int N=N1*N2*N3;
	int nt1 = (int)(0.37*N1), nt2 = (int)(0.26*N2), nt3 = (int) (0.18*N3);  
	int jt = M/2;          // check arbitrary choice of one targ pt
	CPX ct = CPX(0,0), J = IMA*(FLT)iflag*(FLT)scale;
	int m=0;
	for (int m3=-(N3/2); m3<=(N3-1)/2; ++m3){  // loop in correct order over F
		for (int m2=-(N2/2); m2<=(N2-1)/2; ++m2){  // loop in correct order over F
			for (int m1=-(N1/2); m1<=(N1-1)/2; ++m1){
				if(dim==2)
					ct += fk[m++] * exp(J*(m1*x[jt] + m2*y[jt]));   // crude direct
				if(dim==3)
					ct += fk[m++] * exp(J*(m1*x[jt] + m2*y[jt] + m3*z[jt]));   // crude direct
			}
		}
	}
	printf("[gpu   ] one targ: rel err in c[%ld] is %.3g\n",(int64_t)jt,abs(c[jt]-ct)/infnorm(M,c));
	printf("[gpu   ] one targ: abs err in c[%ld] is %.3g\n",(int64_t)jt,abs(c[jt]-ct));
}
