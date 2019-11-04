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

	simple_test_cunfft_2d_Ad(nupts_distr, dim, N1, N2, N3, M);
	//simple_test_cunfft_2d(nupts_distr, N1, N2, M);

	return EXIT_SUCCESS;

}

void simple_test_cunfft_2d_Ad(int nupts_distr,int dim, int N1, int N2, int N3, 
		int M)
{
	resetDevice();
	printf("[info   ] (N1,N2,N3)=(%d,%d,%d), M=%d, spreadwidth=%d\n", 
			N1,N2,N3,M,2*CUT_OFF+2);

	uint_t N[3];
	N[0]=N1;
	N[1]=N2;
	N[2]=N3;

	CNTime timer; timer.start();
	CNTime datatimer;

	cunfft_plan p;
	cunfft_init(&p,dim,N,M);

	/* create random data */
	datatimer.start();
	create_data_type1(nupts_distr, dim, M, &p.x[0], &p.x[1], &p.x[2], 
			dim, dim, dim, &p.f[0], 0.5);
	double td = datatimer.elapsedsec();

	int numOfRuns=1;
	cunfft_reinitAd(&p);
	copyDataToDeviceAd(&p);
	cunfft_adjoint(&p);
	copyDataToHostAd(&p);
	double ti=timer.elapsedsec();
	printf("[time   ] Spread: \t%.3g s\n", 
			p.CUNFFTTimes.time_CONV/numOfRuns);
	printf("[time   ] FFT: \t\t%.3g s\n", 
			p.CUNFFTTimes.time_FFT/numOfRuns);
	printf("[time   ] Deconvolve: \t%.3g s\n", 
			p.CUNFFTTimes.time_ROC/numOfRuns);
	printf("[time   ] Total time = %.3g s\n", ti-td);
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

void simple_test_cunfft_2d(int nupts_distr, int dim, int N1, int N2, int N3, 
		int M)
{
	resetDevice();
	uint_t N[3];
	N[0]=N1;N[1]=N2;
	cunfft_plan p;
	int numOfRuns=1;
	cunfft_init(&p,2,N,M);
	//getExampleData_uniDistr(&p);
	printf("[info  ] cut off = %d", CUT_OFF);

#if 1	
	copyDataToDevice_g(&p);
	cudaVerify(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	double t=cuConvolution(&p);
	p.CUNFFTTimes.time_CONV=t;
	copyDataToHost(&p);
	printf("\n[time  ] cnufft Copy memory HtoD\t %.3g ms\n", 1000*p.CUNFFTTimes.time_COPY_IN);
	printf("[time  ] cnufft interp \t\t\t %.3g ms\n", 1000*p.CUNFFTTimes.time_CONV);
	printf("[time  ] cnufft Copy memory DtoH\t %.3g ms\n", 1000*p.CUNFFTTimes.time_COPY_OUT);
#else	
	copyDataToDevice(&p);
	cunfft_transform(&p);
	copyDataToHost(&p);
	//showCoeff_cuComplex(p.f,32,"vector f (first few entries)");
	showTimes(&p.CUNFFTTimes,numOfRuns);
#endif
	//showCoeff_cuComplex(p.g,2*N1,"vector g (first few entries)");
	cunfft_finalize(&p);
}
