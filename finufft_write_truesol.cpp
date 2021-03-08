// this is all you must include for the finufftf lib...
#include <finufft.h>
#include "create_data.cpp"
#include "utils.h"

// also needed for this example...
#include <stdio.h>
#include <stdlib.h>


using namespace std;

int main(int argc, char *argv[]){
	int ier;
	int N1, N2, N3, M, N;
	if (argc<4) {
		fprintf(stderr,"Usage: finufftf [nupts_distr [dim [N1 N2 N3 [M [tol]]]]\n");
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

	float tol=6e-8;// desired accuracy

	printf("[info  ] (N1,N2,N3)=(%d,%d,%d), M=%d, tol=%3.1e\n", N1,N2,N3,M,tol);

	nufft_opts opts; 
	finufftf_default_opts(&opts);
	if(N3 == 1){
		opts.upsampfac = 2;// some timing results
	}
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
	float totaltime=0;
	CNTime timer; timer.start();
	finufftf_plan plan;
	ier = finufftf_makeplan(type,dim,nmodes,+1,ntrans,tol,&plan,&opts);
	finufftf_setpts(plan,M,x,y,z,0,NULL,NULL,NULL);
	finufftf_execute(plan,c,F);
	finufftf_destroy(plan);
	write_sol(type, nupts_distr, dim, N1, N2, N3, M, c, F);

	memset(c, 0, sizeof(complex<float>)*M);
	type = 2;
	int n[3];
	n[0] = N1;
	n[1] = N2;
	n[2] = N3;
	create_data_type2(nupts_distr, dim, M, x, y, z, 1, 1, 1, n, F, M_PI, N1, N2, N3);
	ier = finufftf_makeplan(type,dim,nmodes,-1,ntrans,tol,&plan,&opts);
	finufftf_setpts(plan,M,x,y,z,0,NULL,NULL,NULL);
	finufftf_execute(plan,c,F);
	finufftf_destroy(plan);
	write_sol(type, nupts_distr, dim, N1, N2, N3, M, c, F);

	return ier;

}
