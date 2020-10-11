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

	float tol=1e-6;
	if(argc>7){
		sscanf(argv[7],"%lf",&w); tol  = (float)w;
	}
	cout<<scientific<<setprecision(3);

	int ntransf = 1;
	int ntransfcufftplan = 1;
	int iflag=1;

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

	float *x, *y, *z;
	complex<float> *c, *fk;
	cudaMallocHost(&x, M*sizeof(float));
	cudaMallocHost(&y, M*sizeof(float));
	cudaMallocHost(&z, M*sizeof(float));
	cudaMallocHost(&c, M*ntransf*sizeof(cuFloatComplex));
	cudaMallocHost(&fk,N1*N2*N3*ntransf*sizeof(cuFloatComplex));

	float *d_x, *d_y, *d_z;
	cuFloatComplex *d_c, *d_fk;
	cudaEventRecord(start);
 	{
		cudaMalloc(&d_x,M*sizeof(float));
		cudaMalloc(&d_y,M*sizeof(float));
		cudaMalloc(&d_z,M*sizeof(float));
		cudaMalloc(&d_c,M*ntransf*sizeof(cuFloatComplex));
		cudaMalloc(&d_fk,N1*N2*N3*ntransf*sizeof(cuFloatComplex));
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	gpumemtime+=milliseconds;

	create_data_type2(nupts_distr, dim, M, x, y, z, 1, 1, 1, nmodes, fk, M_PI, N1, N2, N3);

	cudaEventRecord(start);
 	{
		cudaMemcpy(d_x,x,M*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(d_y,y,M*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(d_z,z,M*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(d_fk,fk,N1*N2*N3*ntransf*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	gpumemtime+=milliseconds;

	cufinufftf_plan dplan;
	cufinufft_opts opts;
	ier=cufinufft_default_opts(2, dim, &opts);

	opts.gpu_method=1;
	opts.gpu_kerevalmeth=1;
	opts.gpu_sort=1;

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
		ier=cufinufftf_makeplan(2, dim, nmodes, iflag, ntransf, tol, 
				ntransfcufftplan, &dplan, &opts);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft plan:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	{
		ier=cufinufftf_setpts(M, d_x, d_y, d_z, 0, NULL, NULL, NULL, dplan);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft setNUpts:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	{
		ier=cufinufftf_execute(d_c, d_fk, dplan);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	totaltime += milliseconds;
	printf("[time  ] cufinufft exec:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
	{
		ier=cufinufftf_destroy(dplan);
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
		cudaMemcpy(c,d_c,M*ntransf*sizeof(cuFloatComplex),cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	gpumemtime+=milliseconds;
	printf("[time  ] total+gpumem: %.3g s\n", (totaltime+gpumemtime)/1000);

#ifdef ACCURACY
	accuracy_check_type2(dim, iflag, N1, N2, N3, M, x, y, z, 1, 1, 1, c, fk, 1.0);
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
