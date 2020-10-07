/*
 * simpleTest.cpp
 * Examples for usinf CUNFFT lib
 *  Created on: 23.07.2015
 *      Author: sukunis
 */

#include <stdarg.h>
#include <float.h>
#include <unistd.h> // get current dir
#include <cunfft_util.h>
#include <cunfft.h>
#include <cunfft_kernel.cuh>

#include <limits.h>
#include <float.h>
#include "utils.h"
#include "create_data.cpp"


void simple_test_cunfft_2d_Ad(int nupts_distr,int dim, int N1, int N2, int N3, 
	int M);
int main(int argc, char** argv)
{
	if (argc<4) {
		fprintf(stderr,"Usage: cunfft [nupts_distr [dim [N1 N2 N3 [M]]\n");
		return 1;
	}
	int dim=3;
	int N1, N2, N3;
	double w;
	int nupts_distr;
	sscanf(argv[1],"%d",&nupts_distr);
	sscanf(argv[2],"%d",&dim);
	sscanf(argv[3],"%lf",&w); N1 = (int)w;  // so can read 1e6 right!
	sscanf(argv[4],"%lf",&w); N2 = (int)w;  // so can read 1e6 right!
	sscanf(argv[5],"%lf",&w); N3 = (int)w;  // so can read 1e6 right!

	int M;
	M = 8*N1*N2*N3;// let density always be 1
	if(argc>6){
		sscanf(argv[6],"%lf",&w); M  = (int)w;  // so can read 1e6 right!
	}

	float tol=1e-6;
	if(argc>7){
		sscanf(argv[7],"%lf",&w); tol  = (float)w;
	}

	int ns = std::ceil(-log10(tol/10.0));;
	printf("ns = %d\n", ns);
	if(2*CUT_OFF+2 != ns){
		printf("2CUTOFF+2 is not equal to ns\n");
		return 0;
	}

	simple_test_cunfft_2d_Ad(nupts_distr, dim, N1, N2, N3, M);

	return EXIT_SUCCESS;

}

void simple_test_cunfft_2d_Ad(int nupts_distr,int dim, int N1, int N2, int N3, 
		int M)
{
	resetDevice();

	uint_t N[3];
	N[0]=N1;
	N[1]=N2;
	N[2]=N3;

	CNTime timer; 
	timer.start();

	cunfft_plan p;
	cunfft_init(&p,dim,N,M);

	/* create random data */
	create_data_type1(nupts_distr, dim, M, &p.x[0], &p.x[1], &p.x[2], 
			dim, dim, dim, &p.f[0], 0.5, N1, N2, N3);

	GPUTimer t=getTimeGPU();
	int numOfRuns=1;
	copyDataToDeviceAd(&p);
	t=getTimeGPU();
	cunfft_adjoint(&p);
	double runTime = elapsedGPUTime(t,getTimeGPU());
	copyDataToHostAd(&p);
	p.CUNFFTTimes.runTime=runTime;

	printf("[time   ] spread: \t%.3g s\n",     p.CUNFFTTimes.time_CONV/numOfRuns);
	printf("[time   ] fft: \t\t%.3g s\n",      p.CUNFFTTimes.time_FFT/numOfRuns);
	printf("[time   ] deconvolve: \t%.3g s\n", p.CUNFFTTimes.time_ROC/numOfRuns);
	printf("[time   ] exec: \t%.3g s\n",      (p.CUNFFTTimes.time_ROC+p.CUNFFTTimes.time_FFT+p.CUNFFTTimes.time_CONV)/numOfRuns);
	printf("[time   ] Totaltime: \t%.3g s\n", p.CUNFFTTimes.runTime);
	/*
	//CUNFFT adjoint
	cunfft_reinitAd(&p);
	copyDataToDeviceAd(&p);

	cudaVerify(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	double t=0.0;
	t=cuConvolution_adjoint(&p);
	copy_g_ToHost(&p);
	printf("\n[time  ] cnufft Copy memory HtoD\t %.3g ms\n", 1000*p.CUNFFTTimes.time_COPY_IN);
	printf("[time  ] cnufft spread \t\t\t %.3g ms\n", 1000*t);
	printf("[time  ] cnufft Copy memory DtoH\t %.3g ms\n", 1000*p.CUNFFTTimes.time_COPY_OUT);
	//showCoeff_cuComplex(p.g,2*N1,"vector g (first few entries)");
	*/
	cunfft_finalize(&p);
}
