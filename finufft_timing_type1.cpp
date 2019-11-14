// this is all you must include for the finufft lib...
#include <finufft.h>
#include "utils.h"

// also needed for this example...
#include <stdio.h>
#include <stdlib.h>

#include "create_data.cpp"

int main(int argc, char *argv[]){
	int ier;
	int N1, N2, N3, M, N;
	if (argc<4) {
		fprintf(stderr,"Usage: finufft [nupts_distr [dim [N1 N2 N3 [M [tol]]]]\n");
		return 1;
	}  
	double w;
	int nupts_distr;
	int dim;
	sscanf(argv[1],"%d" ,&nupts_distr);
	sscanf(argv[2],"%d" ,&dim);
	sscanf(argv[3],"%lf",&w); N1 = (int)w;  // so can read 1e6 right!
	sscanf(argv[4],"%lf",&w); N2 = (int)w;
	sscanf(argv[5],"%lf",&w); N3 = (int)w;
	N = N1*N2*N3;
	M = 8*N1*N2*N3;// let density always be 1
	if(argc>6){
		sscanf(argv[6],"%lf",&w); M  = (int)w;
	}

	FLT tol=1e-6;// desired accuracy
	if(argc>7){
		sscanf(argv[7],"%lf",&w); tol  = (FLT)w;
	}

	printf("[info  ] (N1,N2,N3)=(%d,%d,%d), M=%d, tol=%3.1e\n", N1,N2,N3,M,tol);

	nufft_opts opts; 
	finufft_default_opts(&opts);
	opts.debug = 2;// some timing results

	FLT *x, *y, *z;
	CPX *c, *F;  	

	x = (FLT *)malloc(sizeof(FLT)*M);
	if(dim>1)
		y = (FLT *)malloc(sizeof(FLT)*M);
	if(dim>2)
		z = (FLT *)malloc(sizeof(FLT)*M);

	c = (CPX*)malloc(sizeof(CPX)*M);
	F = (CPX*)malloc(sizeof(CPX)*N);

	create_data_type1(nupts_distr, dim, M, x, y, z, 1, 1, 1, c, M_PI);
	CNTime timer; timer.start();
	switch(dim){
		case 1:
			ier = finufft1d1(M,&x[0],&c[0],1,tol,N1,&F[0],opts);
			break;
		case 2:
			ier = finufft2d1(M,&x[0],&y[0],&c[0],1,tol,N1,N2,&F[0],opts);
			break;
		case 3:
			ier = finufft3d1(M,&x[0],&y[0],&z[0],&c[0],1,tol,N1,N2,N3,&F[0],opts);
			break;
		default:
			fprintf(stderr, "Invalid dimension\n");
	}
	double ti=timer.elapsedsec();
	printf("[time  ] Total time = %.3g s\n", ti);

	/*
	   double n_x = round(0.45*N1); //check the answer for this arbitrary mode
	   double n_y = round(-0.35*N2);

	   complex<double> Ftest(0,0);
	   for(int j = 0; j < M; j++){
	   Ftest += c[j]*exp(I*(n_x*x[j]+n_y*y[j]));
	   }

		//indicies in output array for this frequency pair?
		int n_out_x = n_x + (int)N1/2; 
		int n_out_y = n_y + (int)N2/2;
		int indexOut = n_out_x + n_out_y*(N1);  

		//compute inf norm of F 
		double Fmax = 0.0;
		for (int m=0; m<N1*N2; m++) {
		double aF = abs(F[m]);
		if (aF>Fmax) Fmax=aF;
		}

		//compute relative error
		double err = abs(F[indexOut] - Ftest)/Fmax; 
	*/
	return ier;

}
