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
		fprintf(stderr,"Usage: cufinufft_type2 [nupts_distr [dim [N1 N2 N3 [M [tol]]]]\n");
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
	cout<<scientific<<setprecision(3);

	int method=2;
	if(argc>8){
		sscanf(argv[8],"%d",&method);
	}

	int kerevalmeth=0;
	if(argc>9){
		sscanf(argv[9],"%d",&kerevalmeth);
	}

	int ntransf = 1;
	int ntransfcufftplan = 1;
	int iflag=-1;

	int nmodes[3];
	nmodes[0] = N1;
	nmodes[1] = N2;
	nmodes[2] = N3;

	cudaEvent_t start, stop;
	float milliseconds = 0;
	float totaltime = 0;
	float gpumemtime = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	double *x, *y, *z;
	complex<double> *c, *fk;
	cudaMallocHost(&x, M*sizeof(double));
	cudaMallocHost(&y, M*sizeof(double));
	cudaMallocHost(&z, M*sizeof(double));
	cudaMallocHost(&c, M*ntransf*sizeof(cuDoubleComplex));
	cudaMallocHost(&fk,N1*N2*N3*ntransf*sizeof(cuDoubleComplex));

	double *d_x, *d_y, *d_z;
	cuDoubleComplex *d_c, *d_fk;
	cudaEventRecord(start);
 	{
		cudaMalloc(&d_x,M*sizeof(double));
		cudaMalloc(&d_y,M*sizeof(double));
		cudaMalloc(&d_z,M*sizeof(double));
		cudaMalloc(&d_c,M*ntransf*sizeof(cuDoubleComplex));
		cudaMalloc(&d_fk,N1*N2*N3*ntransf*sizeof(cuDoubleComplex));
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	gpumemtime+=milliseconds;

	create_data_type2(nupts_distr, dim, M, x, y, z, 1, 1, 1, nmodes, fk, M_PI, N1, N2, N3);

	cudaEventRecord(start);
 	{
		cudaMemcpy(d_x,x,M*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(d_y,y,M*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(d_z,z,M*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(d_fk,fk,N1*N2*N3*ntransf*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	gpumemtime+=milliseconds;

	cufinufft_plan dplan;
	cufinufft_opts opts;
	ier=cufinufft_default_opts(2, dim, &opts);

	opts.gpu_method=1;
	opts.gpu_kerevalmeth=kerevalmeth;
	opts.gpu_sort=1;

	if(dim==3){
		opts.gpu_binsizex=16;
		opts.gpu_binsizey=8;
		opts.gpu_binsizez=4;
		opts.gpu_maxsubprobsize=1024;
	}

	int ns = std::ceil(-log10(tol/10.0));//spread width
	printf("[info  ] (N1,N2,N3)=(%d,%d,%d), M=%d, tol=%3.1e, spreadwidth=%d\n", 
			N1,N2,N3,M,tol,ns);

	CNTime timer; timer.start();
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

	cudaEventRecord(start);
	{
		ier=cufinufft_makeplan(2, dim, nmodes, iflag, ntransf, tol, 
				ntransfcufftplan, &dplan, &opts);
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
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft setNUpts:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	{
		ier=cufinufft_execute(d_c, d_fk, dplan);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft exec:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	{
		ier=cufinufft_destroy(dplan);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds/1000);

	double ti=timer.elapsedsec();
	printf("[time  ] total: %.3g s\n", totaltime/1000);

	cudaEventRecord(start);
	{
		cudaMemcpy(c,d_c,M*ntransf*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	gpumemtime+=milliseconds;
	printf("[time  ] total+gpumem: %.3g s\n", (totaltime+gpumemtime)/1000);

#ifdef ACCURACY
	double err;
	int type=2;
	err = calerr(1, type, nupts_distr, dim, N1, N2, N3, M, c, fk);
	printf("[acc   ] releativeerr: %.3g\n", err);

	accuracy_check_type2(1, dim, iflag, N1, N2, N3, M, x, y, z, 1, 1, 1, c, fk, 1.0);
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
