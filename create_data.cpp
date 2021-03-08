#include <cuComplex.h>
#include <Eigen/Dense>
#include <iostream>
#include <stdio.h>
#include <vector>       // std::vector
#include <ctime>        // std::time
#include <cstdlib> 

#undef FLT
#undef CPX
#undef IMA

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

#include "contrib/dirft2d.cpp"
#include "contrib/dirft3d.cpp"
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
void gauss(int N, FLT *x){
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
				srand (0);
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
  				std::random_shuffle ( vecshuff.begin(), vecshuff.end() );
				
				int m = ceil(pow(M, 1.0/dim));
				if(m%2==1) m=m+1;
				FLT* mug = (FLT*) malloc(m*sizeof(FLT));

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
					FLT* pg  = (FLT*) malloc(m*sizeof(FLT));
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
					FLT* pg  = (FLT*) malloc(mp*sizeof(FLT));
					for(int i=0; i<mp; i++){
						pg[i]  = 2*M_PI*(i+1)/mp;
					}
					int mr = m/2; 
					FLT* rg = (FLT*) malloc(mr*sizeof(FLT));
					gauss(mr, rg);
					for(int i=0; i<mr; i++){
						rg[i] = scale*(1+rg[i])/2.0;
					}
					FLT* sthg = (FLT*) malloc(m*sizeof(FLT));
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
		c[i] = std::complex<FLT>(1.0,1.0);
	}
}

void create_data_type2(int nupts_distr, int dim, int M, FLT* x, FLT* y, FLT* z, 
		int ix, int iy, int iz, int* Nmodes, CPX* f, FLT scale, int N1, 
        int N2, int N3)
{
	create_nupts(nupts_distr, dim, M, x, y, z, ix, iy, iz, scale, N1, N2, N3);
	for (int i = 0; i < Nmodes[0]*Nmodes[1]*Nmodes[2]; i++) {
		f[i] = std::complex<FLT>(1.0,1.0);
	}
}

void print_solution_type1(int lib, int N1, int N2, int N3, CPX* f)
{
	int idx;
	for(int k=0; k<N3; k++){
		for(int j=0; j<N2; j++){
			for(int i=0; i<N1; i++){
				if(lib==3){
					if (N3==1)
						idx = j+i*N2;
					else
						idx = k+j*N3+i*N3*N2;
				}else{
					idx = i+j*N1+k*N1*N2;
				}
				printf("(%3d, %3d, %3d, %10.4f + %10.4fi)\n", i, j, k, f[idx].real(), f[idx].imag());
			}
		}
	}	
}

/* Complex util from FINUFFT src/utils.cpp*/
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

FLT relerrtwonorm(int n, CPX* a, CPX* b)
// ||a-b||_2 / ||a||_2
{
  FLT err = 0.0, nrm = 0.0;
  for (int m=0; m<n; ++m) {
    nrm += real(conj(a[m])*a[m]);
    CPX diff = a[m]-b[m];
    err += real(conj(diff)*diff);
  }
  return sqrt(err/nrm);
}

FLT relerrtwonorm_cunfft(int n1, int n2, int n3, CPX* ftrue, CPX* fapprox)
// ||a-b||_2 / ||a||_2
{
  FLT err = 0.0, nrm = 0.0;
  for (int k=0; k<n3; k++){
    for (int j=0; j<n2; j++){
      for (int i=0; i<n3; i++){
        int idxc = k+j*n3+i*n3*n2;
        int idx = i+j*n1+k*n1*n2;
        nrm += real(conj(ftrue[idx])*ftrue[idx]);
        CPX diff = ftrue[idx]-fapprox[idxc];
        err += real(conj(diff)*diff);
      }
    }
  }
  return sqrt(err/nrm);
}


void accuracy_check_type1(int lib, int dim, int iflag, int N1, int N2, int N3, int M, FLT* x, 
	FLT* y, FLT *z, int ix, int iy, int iz, CPX* c, CPX* fk, FLT scale)
{
	int N=N1*N2*N3;
	int nt1 = (int)(0.37*N1), nt2 = (int)(0.26*N2), nt3 = (int) (0.18*N3);  
	// choose some mode index to check
	CPX Ft = CPX(0,0), J = IMA*(FLT)iflag*(FLT)scale;
	for (int j=0; j<M; ++j){
		if (dim == 2)
			Ft += c[j] * exp(J*(nt1*x[j*ix]+nt2*y[j*iy]));   // crude direct
		if (dim == 3)
			Ft += c[j] * exp(J*(nt1*x[j*ix]+nt2*y[j*iy]+nt3*z[j*iz]));   // crude direct
	}
	int it;
	if(dim==2){
		if(lib == 3) // CUNFFT
			it = N1/2+nt2 + N1*(N2/2+nt1);
		else // cuFINUFFT, FINUFFT
			it = N1/2+nt1 + N1*(N2/2+nt2);
		printf("[acc check] one mode: rel err in F[%ld,%ld] is %.3g\n",(int)nt1,
				(int)nt2,abs(Ft-fk[it])/infnorm(N,fk));
		printf("[acc check] one mode: abs err in F[%ld,%ld] is %.3g\n",(int)nt1,
				(int)nt2,abs(Ft-fk[it]));
	}
	if(dim==3){
		if(lib == 3)
			it = N3/2+nt3 + N3*(N2/2+nt2) + N3*N2*(N1/2+nt1);   // index in complex F as 1d array
		else
			it = N1/2+nt1 + N1*(N2/2+nt2) + N1*N2*(N3/2+nt3);   // index in complex F as 1d array
		printf("[acc check] one mode: rel err in F[%ld,%ld,%ld] is %.3g\n",
				(int)nt1,(int)nt2,(int)nt3, abs(Ft-fk[it])/infnorm(N,fk));
		printf("[acc check] one mode: abs err in F[%ld,%ld,%ld] is %.3g\n",
				(int)nt1,(int)nt2,(int)nt3, abs(Ft-fk[it]));
	}
	if( N<1e8){
		CPX* Ft = (CPX*)malloc(sizeof(CPX)*N);
		if(dim==2)
			dirft2d1(lib,M,x,y,ix,iy,c,iflag,N1,N2,Ft);
		if(dim==3)
			dirft3d1(lib,M,x,y,z,ix,iy,iz,c,iflag,N1,N2,N3,Ft);
		FLT err = relerrtwonorm(N,Ft,fk);
		printf("[acc check] dirft: rel l2-err of result F is %.3g\n",err);
		free(Ft);
	}
}

void accuracy_check_type2(int lib, int dim, int iflag, int N1, int N2, int N3, 
    int M, FLT* x, FLT* y, FLT *z, int ix, int iy, int iz, CPX* c, CPX* fk, FLT scale)
{
	// Check one non-uniform point
	int N=N1*N2*N3;
	int nt1 = (int)(0.37*N1), nt2 = (int)(0.26*N2), nt3 = (int) (0.18*N3);  
	int jt = M/2;          // check arbitrary choice of one targ pt
	CPX ct = CPX(0,0), J = IMA*(FLT)iflag*(FLT)scale;
	if(lib==3){
		int m=0;
		for (int m1=-(N1/2); m1<=(N1-1)/2; ++m1){
			for (int m2=-(N2/2); m2<=(N2-1)/2; ++m2){  // loop in correct order over F
				for (int m3=-(N3/2); m3<=(N3-1)/2; ++m3){  // loop in correct order over F
					if(dim==2)
						ct += fk[m++] * exp(J*(m1*x[jt*ix] + m2*y[jt*iy]));   // crude direct
					if(dim==3)
						ct += fk[m++] * exp(J*(m1*x[jt*ix] + m2*y[jt*iy] + m3*z[jt*iz]));   // crude direct
				}
			}
		}
	}else{
		int m=0;
		for (int m3=-(N3/2); m3<=(N3-1)/2; ++m3){  // loop in correct order over F
			for (int m2=-(N2/2); m2<=(N2-1)/2; ++m2){  // loop in correct order over F
				for (int m1=-(N1/2); m1<=(N1-1)/2; ++m1){
					if(dim==2)
						ct += fk[m++] * exp(J*(m1*x[jt*ix] + m2*y[jt*iy]));   // crude direct
					if(dim==3)
						ct += fk[m++] * exp(J*(m1*x[jt*ix] + m2*y[jt*iy] + m3*z[jt*iz]));   // crude direct
				}
			}
		}
	}
	printf("[acc check] one targ: rel err in c[%ld] is %.3g\n",(int64_t)jt,abs(c[jt]-ct)/infnorm(M,c));
	printf("[acc check] one targ: abs err in c[%ld] is %.3g\n",(int64_t)jt,abs(c[jt]-ct));

	if(N<1e8 && M<1e6){
		CPX* Ct = (CPX*)malloc(sizeof(CPX)*M);
		if(dim==2)
			dirft2d2(lib,M,x,y,ix,iy,Ct,iflag,N1,N2,fk);
		if(dim==3)
			dirft3d2(lib,M,x,y,z,ix,iy,iz,Ct,iflag,N1,N2,N3,fk);
		FLT err = relerrtwonorm(M,Ct,c);
		printf("[acc check] dirft: rel l2-err of result c is %.3g\n",err);
	}
}

void write_sol(int type, int nupts, int dim, int N1, int N2, int N3, int M, CPX* c, CPX* fk)
{
	char filename[100];
	FILE * pFile;

	if(type==1){
		if(dim==2){
			sprintf(filename, "true_2d1_nupts%d_N%d_M1e7_%d.bin", nupts, N1, sizeof(FLT));
			pFile = fopen (filename, "wb");
		}
		if(dim==3){
			sprintf(filename, "true_3d1_nupts%d_N%d_M1e7_%d.bin", nupts, N1, sizeof(FLT));
			pFile = fopen (filename, "wb");
		}
		fwrite (fk, sizeof(complex<FLT>), N1*N2*N3, pFile);
	}

	if(type==2){
		if(dim==2){
			sprintf(filename, "true_2d2_nupts%d_N%d_M1e7_%d.bin", nupts, N1, sizeof(FLT));
			pFile = fopen (filename, "wb");
		}
		if(dim==3){
			sprintf(filename, "true_3d2_nupts%d_N%d_M1e7_%d.bin", nupts, N1, sizeof(FLT));
			pFile = fopen (filename, "wb");
		}
		fwrite (c, sizeof(complex<FLT>), M, pFile);
	}
}

FLT calerr(int lib, int type, int nupts, int dim, int N1, int N2, int N3, int M, CPX* c, CPX* fk)
{
	char filename[100];
	FILE * pFile;
	FLT err;
	
	if(type==1){
		complex<FLT> *fktrue=(complex<FLT>*)malloc(N1*N2*N3*sizeof(complex<FLT>));
		if(dim==2){
			sprintf(filename, "true_2d1_nupts%d_N%d_M1e7_%d.bin", nupts, N1, sizeof(FLT));
			pFile = fopen (filename, "rb");
		}
		if(dim==3){
			sprintf(filename, "true_3d1_nupts%d_N%d_M1e7_%d.bin", nupts, N1, sizeof(FLT));
			pFile = fopen (filename, "rb");
		}
		fread (fktrue, sizeof(complex<FLT>), N1*N2*N3, pFile);
		if(lib == 3){
			err = relerrtwonorm_cunfft(N1,N2,N3,fktrue,fk);
		}else{
			err = relerrtwonorm(N1*N2*N3,fktrue,fk);
		}
	}

	if(type==2){
		complex<FLT> *ctrue=(complex<FLT>*)malloc(M*sizeof(complex<FLT>));
		if(dim==2){
			sprintf(filename, "true_2d2_nupts%d_N%d_M1e7_%d.bin", nupts, N1, sizeof(FLT));
			pFile = fopen (filename, "rb");
		}
		if(dim==3){
			sprintf(filename, "true_3d2_nupts%d_N%d_M1e7_%d.bin", nupts, N1, sizeof(FLT));
			pFile = fopen (filename, "rb");
		}
		fread (ctrue, sizeof(complex<FLT>), M, pFile);
		err = relerrtwonorm(M,ctrue,c);
	}
	return err;
}
