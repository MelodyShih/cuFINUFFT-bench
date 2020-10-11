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


void simple_test_cunfft_2d(int nupts_distr,int dim, int N1, int N2, int N3, 
	int M);
int main(int argc, char** argv)
{
	if (argc<4) {
		fprintf(stderr,"Usage: cunfft_type2 [nupts_distr [dim [N1 N2 N3 [M]]\n");
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
	if(2*CUT_OFF+2 != ns){
		printf("2CUTOFF+2 is not equal to ns\n");
		return 0;
	}
	simple_test_cunfft_2d(nupts_distr, dim, N1, N2, N3, M);

	return EXIT_SUCCESS;

}

void simple_test_cunfft_2d(int nupts_distr,int dim, int N1, int N2, int N3, 
		int M)
{
	resetDevice();
	printf("[info   ] (N1,N2,N3)=(%d,%d,%d), M=%d, spreadwidth=%d\n", 
			N1,N2,N3,M,2*CUT_OFF+2);

	uint_t N[3];
	N[0]=N1;
	N[1]=N2;
	N[2]=N3;

	int Nmodes[3];
	Nmodes[0]=N1;
	Nmodes[1]=N2;
	Nmodes[2]=N3;

	{
		cufftHandle fftplan;
	        int nf2=1;
        	int nf1=1;
	    cufftPlan1d(&fftplan,nf1,CUFFT_C2C,1);
	}

	cunfft_plan p;
	GPUTimer t=getTimeGPU();
	cunfft_init(&p,dim,N,M);
	double totalgpumem = elapsedGPUTime(t,getTimeGPU());

	/* create random data */
	create_data_type2(nupts_distr, dim, M, &p.x[0], &p.x[1], &p.x[2], dim, dim, 
					  dim, Nmodes, (std::complex<float>*) &p.f_hat[0], 0.5, N1, N2, N3);

	t=getTimeGPU();
	copyDataToDevice(&p);
	totalgpumem += elapsedGPUTime(t,getTimeGPU());

	t=getTimeGPU();
	cunfft_transform(&p);
	double exec= elapsedGPUTime(t,getTimeGPU());

	t=getTimeGPU();
	copyDataToHost(&p);
	totalgpumem += elapsedGPUTime(t,getTimeGPU());

	printf("[time   ] unspread: \t%.3g s\n", p.CUNFFTTimes.time_CONV);
	printf("[time   ] fft: \t\t%.3g s\n",    p.CUNFFTTimes.time_FFT);
	printf("[time   ] convolve: \t%.3g s\n", p.CUNFFTTimes.time_ROC);
	printf("[time   ] exec: %.3g s\n", exec);
	printf("[time   ] total+gpumem: \t%.3g s\n", totalgpumem);

#ifdef ACCURACY
	accuracy_check_type2(dim, -1, N1, N2, N3, M, 
						 &p.x[0], &p.x[1], &p.x[2],dim, dim, dim, 
				         (std::complex<float> *)&p.f[0], 
						 (std::complex<float> *)&p.f_hat[0], 2*M_PI);
#endif

	cunfft_finalize(&p);
}
