#include <iostream>
#include <iomanip>
#include <math.h>
#include <complex>

#include <cufinufft.h>
#include "create_data.cpp"

using namespace std;

int main(int argc, char* argv[])
{
	int ier;
	int N1, N2, N3, M, N;
	if (argc<4) {
		fprintf(stderr,"Usage: cufinufft [nupts_distr [dim [N1 N2 N3 [M [tol [method [kerevalmeth]]]]]]\n");
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

	M = 8*N1*N2*N3;// let density always be 1
	if(argc>6){
		sscanf(argv[6],"%lf",&w); M  = (int)w;
	}

	double tol=1e-6;
	if(argc>7){
		sscanf(argv[7],"%lf",&w); tol  = (double)w;
	}

	int method=2;
	if(argc>8){
		sscanf(argv[8],"%d",&method);
	}

	int kerevalmeth=0;
	if(argc>9){
		sscanf(argv[9],"%d",&kerevalmeth);
	}

	cout<<scientific<<setprecision(3);

	int ntransf = 1;
	int maxbatchsize = 1;
	int iflag=1;

	cudaEvent_t start, stop;
	float milliseconds = 0;
	float totaltime = 0;
	float gpumemtime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	double *x, *y, *z;
	complex<double> *c, *fk;
	cudaMallocHost(&x, M*sizeof(double));
	if(dim > 1)
		cudaMallocHost(&y, M*sizeof(double));
	if(dim > 2)
		cudaMallocHost(&z, M*sizeof(double));
	cudaMallocHost(&c, M*ntransf*sizeof(cuDoubleComplex));
	cudaMallocHost(&fk,N1*N2*N3*ntransf*sizeof(cuDoubleComplex));

	double *d_x, *d_y, *d_z;
	cuDoubleComplex *d_c, *d_fk;
	cudaEventRecord(start);
 	{
		cudaMalloc(&d_x,M*sizeof(double));
		if(dim > 1)
			cudaMalloc(&d_y,M*sizeof(double));
		if(dim > 2)
			cudaMalloc(&d_z,M*sizeof(double));
		cudaMalloc(&d_c,M*ntransf*sizeof(cuDoubleComplex));
		cudaMalloc(&d_fk,N1*N2*N3*ntransf*sizeof(cuDoubleComplex));
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	gpumemtime+=milliseconds;

	create_data_type1(nupts_distr, dim, M, x, y, z, 1, 1, 1, c, M_PI, N1, N2, N3);

	cudaEventRecord(start);
 	{
		cudaMemcpy(d_x,x,M*sizeof(double),cudaMemcpyHostToDevice);
		if(dim > 1)
			cudaMemcpy(d_y,y,M*sizeof(double),cudaMemcpyHostToDevice);
		if(dim > 2)
			cudaMemcpy(d_z,z,M*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(d_c,c,M*ntransf*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	gpumemtime+=milliseconds;

	cufinufft_plan dplan;
	cufinufft_opts opts;
	ier=cufinufft_default_opts(1, dim, &opts);
	opts.gpu_method=method;
	opts.gpu_sort=1;
	opts.gpu_kerevalmeth=kerevalmeth;
	if(N1==2048){
		opts.gpu_binsizex=64;
		opts.gpu_binsizey=64;
	}
	if(dim==3){
		opts.gpu_binsizex=16;
		opts.gpu_binsizey=8;
		opts.gpu_binsizez=4;
		opts.gpu_maxsubprobsize=1024;
	}

	int nmodes[3];
	nmodes[0] = N1;
	nmodes[1] = N2;
	nmodes[2] = N3;

	cudaEventRecord(start);
 	{
		cufftHandle fftplan;
	        int nf2=1;
        	int nf1=1;
        	int n[] = {nf2, nf1};
        	int inembed[] = {nf2, nf1};
	    cufftPlan1d(&fftplan,nf1,CUFFT_TYPE,1);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] dummy warmup call to CUFFT\t %.3g s\n", milliseconds/1000);
	CNTime timer; timer.start();
	cudaEventRecord(start);
	{
		ier=cufinufft_makeplan(1, dim, nmodes, iflag, ntransf, tol,
			       maxbatchsize, &dplan, &opts);
		if(ier > 0) return 0;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft plan:\t\t %.3g s\n", milliseconds/1000);
#ifdef ACCURACY
	printf("[acc check] ns=%d\n", dplan->spopts.nspread);
#endif

	cudaEventRecord(start);
	{
		ier=cufinufft_setpts(M, d_x, d_y, d_z, 0, NULL, NULL, NULL, dplan);
		if(ier > 0) return 0;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft setpts:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	{
		ier=cufinufft_execute(d_c, d_fk, dplan);
		if(ier > 0) return 0;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft exec:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	{
		ier=cufinufft_destroy(dplan);
		if(ier > 0) return 0;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds/1000);

	printf("[time  ] total: %.3g s\n", totaltime/1000);
	cudaEventRecord(start);
	{
		cudaMemcpy(fk,d_fk,N1*N2*N3*ntransf*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	gpumemtime+=milliseconds;
	printf("[time  ] total+gpumem: %.3g s\n", (totaltime+gpumemtime)/1000);

#ifdef ACCURACY
	double err;
	int type=1;
	err = calerr(1, type, nupts_distr, dim, N1, N2, N3, M, c, fk);
	printf("[acc   ] releativeerr: %.3g\n", err);

	accuracy_check_type1(1, dim, iflag, N1, N2, N3, M, x, y, z, 1, 1, 1, c, fk, 1.0);
	//print_solution_type1(1, N1, N2, N3, fk);
#endif

	cudaFreeHost(x);
	cudaFreeHost(y);
	cudaFreeHost(z);
	cudaFreeHost(c);
	cudaFreeHost(fk);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
	cudaFree(d_c);
	cudaFree(d_fk);
	return 0;
}
