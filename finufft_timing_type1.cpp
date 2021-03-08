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
	opts.debug = 0;// some timing results
	opts.upsampfac = 2;
	//opts.nthreads = 28;

	float *x, *y, *z;
	complex<float> *c, *F;  	

	x = (float *)malloc(sizeof(float)*M);
	if(dim>1)
		y = (float *)malloc(sizeof(float)*M);
	if(dim>2)
		z = (float *)malloc(sizeof(float)*M);

	c = (complex<float>*)malloc(sizeof(complex<float>)*M);
	F = (complex<float>*)malloc(sizeof(complex<float>)*N);

	create_data_type1(nupts_distr, dim, M, x, y, z, 1, 1, 1, c, M_PI, N1, N2, N3);

	int64_t nmodes[3];
	nmodes[0] = N1;
	nmodes[1] = N2;
	nmodes[2] = N3;

	int type=1,ntrans=1;
	double totaltime=0;
	CNTime timer; timer.start();
	finufftf_plan plan;
	ier = finufftf_makeplan(type,dim,nmodes,+1,ntrans,tol,&plan,&opts);
	double ti=timer.elapsedsec();
	totaltime += ti;
	printf("[time  ] finufft makeplan: \t%.3g s\n", ti);
#ifdef ACCURACY
	printf("[acc check] ns=%d\n", plan->spopts.nspread);
#endif
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
	printf("[time  ] total: \t\t%.3g s\n", totaltime);

#ifdef ACCURACY
	float err;
	err = calerr(0, type, nupts_distr, dim, N1, N2, N3, M, c, F);
	printf("[acc   ] releativeerr: %.3g\n", err);

	accuracy_check_type1(0, dim, +1, N1, N2, N3, M, x, y, z, 1, 1, 1, c, F, 1.0);
	//print_solution_type1(0, N1, N2, N3, F);
#endif


	return ier;

}
