// this is all you must include for the finufft lib...
#include <finufft.h>
#include "utils.h"

// also needed for this example...
#include <stdio.h>
#include <stdlib.h>

#include "create_data.cpp"

using namespace std;

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

	float tol=1e-6;// desired accuracy
	if(argc>7){
		sscanf(argv[7],"%lf",&w); tol  = (float)w;
	}

	printf("[info  ] (N1,N2,N3)=(%d,%d,%d), M=%d, tol=%3.1e\n", N1,N2,N3,M,tol);

	nufft_opts opts; 
	finufft_default_opts(&opts);
	opts.debug = 2;// some timing results

	float *x, *y, *z;
	complex<float> *c, *F;  	

	x = (float *)malloc(sizeof(float)*M);
	if(dim>1)
		y = (float *)malloc(sizeof(float)*M);
	if(dim>2)
		z = (float *)malloc(sizeof(float)*M);

	c = (complex<float>*)malloc(sizeof(complex<float>)*M);
	F = (complex<float>*)malloc(sizeof(complex<float>)*N);

	int n[3];
	n[0] = N1;
	n[1] = N2;
	n[2] = N3;
	create_data_type2(nupts_distr, dim, M, x, y, z, 1, 1, 1, n, F, M_PI, N1, N2, N3);


	int64_t nmodes[3];
	nmodes[0] = N1;
	nmodes[1] = N2;
	nmodes[2] = N3;
	int type=2,ntrans=1;
	double totaltime=0;
	CNTime timer; timer.start();
	finufftf_plan plan;
	ier = finufftf_makeplan(type,dim,nmodes,+1,ntrans,tol,&plan,NULL);
	double ti=timer.elapsedsec();
	totaltime += ti;
	printf("[time  ] finufft makeplan: \t%.3g s\n", ti);
	timer.start();
	finufftf_setpts(plan,M,x,y,z,0,NULL,NULL,NULL);
	ti=timer.elapsedsec();
	totaltime += ti;
	printf("[time  ] finufft setpts: \t%.3g s\n", ti);
	timer.start();
	finufftf_execute(plan,c,F);
	ti=timer.elapsedsec();
	totaltime += ti;
	printf("[time  ] finufft exec: \t\t%.3g s\n", ti);
	timer.start();
	finufftf_destroy(plan);
	ti=timer.elapsedsec();
	totaltime += ti;
	printf("[time  ] finufft destroy: \t%.3g s\n", ti);
	printf("[time  ] Totaltime: \t\t%.3g s\n", totaltime);

#ifdef ACCURACY
	accuracy_check_type2(dim, +1, N1, N2, N3, M, x, y, z, 1, 1, 1, c, F, 1.0);
#endif
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
