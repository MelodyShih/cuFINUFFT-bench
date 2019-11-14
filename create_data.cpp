#include <cuComplex.h>
#include <Eigen/Dense>
#include <iostream>

#ifndef FLT
typedef float FLT;
#endif
#ifndef IMA 
#define IMA std::complex<FLT>(0.0,1.0) 
#endif
#ifndef CUCPX
#ifdef GPU
typedef cuFloatComplex CUCPX;
#else
typedef std::complex<float> CUCPX;
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
void gauss(int N, double *x){
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
		int iy, int iz, FLT scale)
{
	int i,j;
	switch(nupts_distr){
		case 1:
			{
				for (i = 0; i < M; i++) {
					x[i*ix] = scale*randm11();
					if(dim > 1)
						y[i*iy] = scale*randm11();
					if(dim > 2)
						z[i*iz] = scale*randm11();
				}
			}
			break;
		case 2:
			{
				int m = int(pow(M, 1.0/dim));
				if(m%2==1) m=m+1;
				double* mug = (double*) malloc(m*sizeof(double));

				for(i=0; i<m; i++){
					mug[i] = cos(M_PI*((i+1)- 0.5)/m);
				}
				 
				if(dim == 1){
					int n_set = 0;
					for(i=0; i<m; i++){
						if(i > M){
							printf("warning: the nupts are trucated (%d)\n", m*m-n_set);
							return;
						}
						x[i*ix] = scale*mug[i]; 
						n_set++;
					}
				}

				if(dim == 2){
					double* pg  = (double*) malloc(m*sizeof(double));
					for(i=0; i<m; i++){
						pg[i]  = 2*M_PI*(i+1)/m;
					}
					int n_set = 0;
					for(i=0; i<m; i++){
						for(j=0; j<m; j++){
							int idx = i*m+j;
							if(idx > M){
								printf("warning: the nupts are trucated (%d)\n", m*m-n_set);
								return;
							}
							x[idx*ix] = scale*mug[j]*cos(pg[i]);
							y[idx*iy] = scale*mug[j]*sin(pg[i]);
						}
					}
				}
				if(dim == 3){
					int mp = 2*m;
					double* pg  = (double*) malloc(mp*sizeof(double));
					for(i=0; i<mp; i++){
						pg[i]  = 2*M_PI*(i+1)/mp;
					}
					int mr = m/2; 
					double* rg = (double*) malloc(mr*sizeof(double));
					gauss(mr, rg);
					for(i=0; i<mr; i++){
						rg[i] = scale*(1+rg[i])/2.0;
					}
					double* sthg = (double*) malloc(m*sizeof(double));
					for(i=0; i<m; i++){
						sthg[i] = sqrt(1 - mug[i]*mug[i]);
					}
					int n_set = 0;
					for(int r=0; r<mr; r++){
						for(int i=0; i<mp; i++){
							for(int j=0; j<m; j++){
								int idx = r*mp*m + i*m + j;
								if(idx > M){
									printf("warning: the nupts are trucated (%d)\n", m*m*m-n_set);
									return;
								}
								x[idx*ix] = rg[r]*cos(pg[i])*sthg[j];
								y[idx*iy] = rg[r]*sin(pg[i])*sthg[j];
								z[idx*iz] = rg[r]*mug[j];
								n_set++;
							}
						}
					}
					
				}
			}
			break;

		default:
			fprintf(stderr,"Invalid distribution of nonuniform points\n");
	}	
}
void create_data_type1(int nupts_distr, int dim, int M, FLT* x, FLT* y, FLT* z, 
		int ix, int iy, int iz, CUCPX* c, FLT scale)
{
	create_nupts(nupts_distr, dim, M, x, y, z, ix, iy, iz, scale);
	for (int i = 0; i < M; i++) {
#ifdef GPU
		c[i].x = randm11();
		c[i].y = randm11();
#else
		c[i] = crandm11();
#endif
	}
}

void create_data_type2(int nupts_distr, int dim, int M, FLT* x, FLT* y, FLT* z, 
		int ix, int iy, int iz, int* Nmodes, CUCPX* f, FLT scale)
{
	create_nupts(nupts_distr, dim, M, x, y, z, ix, iy, iz, scale);
	for (int i = 0; i < Nmodes[0]*Nmodes[1]*Nmodes[2]; i++) {
#ifdef GPU
		f[i].x = randm11();
		f[i].y = randm11();
#else
		f[i] = crandm11();
#endif
	}
}
