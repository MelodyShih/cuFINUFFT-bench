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
		fprintf(stderr,"Usage: cufinufft [nupts_distr [dim [N1 N2 N3 [M [tol]]]]\n");
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
	
	M = N1*N2*N3;// let density always be 1
	if(argc>6){
		sscanf(argv[6],"%lf",&w); M  = (int)w;
	}

	FLT tol=1e-6;
	if(argc>7){
		sscanf(argv[7],"%lf",&w); tol  = (FLT)w;
	}
	cout<<scientific<<setprecision(3);

	int ntransf = 1;
	int ntransfcufftplan = 1;
	int iflag=1;

    cudaEvent_t start, stop;
    float milliseconds = 0;
    float totaltime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	FLT *x, *y, *z;
	CUCPX *c, *fk;
	cudaMallocHost(&x, M*sizeof(FLT));
	cudaMallocHost(&y, M*sizeof(FLT));
	cudaMallocHost(&z, M*sizeof(FLT));
	cudaMallocHost(&c, M*ntransf*sizeof(CPX));
	cudaMallocHost(&fk,N1*N2*N3*ntransf*sizeof(CPX));

	FLT *d_x, *d_y, *d_z;
	CUCPX *d_c, *d_fk;
	cudaMalloc(&d_x,M*sizeof(FLT));
	cudaMalloc(&d_y,M*sizeof(FLT));
	cudaMalloc(&d_z,M*sizeof(FLT));
	cudaMalloc(&d_c,M*ntransf*sizeof(CUCPX));
	cudaMalloc(&d_fk,N1*N2*N3*ntransf*sizeof(CUCPX));

	create_data_type1(nupts_distr, dim, M, x, y, z, 1, 1, 1, c, M_PI);

	cudaMemcpy(d_x,x,M*sizeof(FLT),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,y,M*sizeof(FLT),cudaMemcpyHostToDevice);
	cudaMemcpy(d_z,z,M*sizeof(FLT),cudaMemcpyHostToDevice);
	cudaMemcpy(d_c,c,M*ntransf*sizeof(CUCPX),cudaMemcpyHostToDevice);

	cufinufft_plan dplan;

	int nmodes[3];

	ier=cufinufft_default_opts(type1, dim, dplan.opts);

	nmodes[0] = N1;
	nmodes[1] = N2;
	nmodes[2] = N3;
	
	int ns = std::ceil(-log10(tol/10.0));//spread width
	printf("[info  ] (N1,N2,N3)=(%d,%d,%d), M=%d, tol=%3.1e, spreadwidth=%d\n", 
		N1,N2,N3,M,tol,ns);

	CNTime timer; timer.start();
	cudaEventRecord(start);
    {
		ier=cufinufft_makeplan(type1, dim, nmodes, iflag, ntransf, tol, 
			ntransfcufftplan, &dplan);
	}
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
    printf("[time  ] cufinufft plan:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
    {
		ier=cufinufft_setNUpts(M, d_x, d_y, d_z, 0, NULL, NULL, NULL, &dplan);
	}
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
	printf("[time  ] cufinufft setNUpts:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
    {
		ier=cufinufft_exec(d_c, d_fk, &dplan);
	}
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
    printf("[time  ] cufinufft exec:\t\t %.3g s\n", milliseconds/1000);

	cudaEventRecord(start);
    {
		ier=cufinufft_destroy(&dplan);
	}
	cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    totaltime += milliseconds;
    printf("[time  ] cufinufft destroy:\t\t %.3g s\n", milliseconds/1000);

	cudaMemcpy(fk,d_fk,N1*N2*N3*ntransf*sizeof(CUCPX),cudaMemcpyDeviceToHost);
	double ti=timer.elapsedsec();
	printf("[time  ] Total time = %.3g s\n", ti);

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
