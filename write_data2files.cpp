#include <limits.h>
#include <stdio.h> 
#include <float.h>
#include "utils.h"
#include "create_data.cpp"

using namespace std;

int main(int argc, char** argv)
{
	int dim=3;
	int N1, N2, N3;
	int M=134217728;

	int N[]={8,16};
	float density[]={1};
	int nupts_distr[] = {1};
	//int N[]={64, 128, 256, 512, 1024, 2048};
	//float density[]={0.1, 1, 10};
	
	//int N[]={32, 64, 128, 256};
	//float density[]={0.1, 1};
	//int nupts_distr[] = {1,2,3};

	char filename[100];

	float*x, *y, *z;
	x = (float*) malloc(M*sizeof(float));
	y = (float*) malloc(M*sizeof(float));
	if(dim==3)
		z = (float*) malloc(M*sizeof(float));

	FILE * pFile;
	for(int i=0; i<1; i++){
		for(int d=0; d<1; d++){
			for(int n=0; n<2; n++){
				//M = int(floor(pow(N[n],dim)*pow(2,dim)*density[d]));
				M = pow(N[n],dim);
				sprintf(filename, "data/DIM_%d_NUPTS_%d_N_%d_M_%d.dat", dim, nupts_distr[i], N[n], M);
				cout<<filename<<endl;
#if 1
  				pFile = fopen(filename,"w");
				create_nupts(nupts_distr[i], dim, M, x, y, z, 1, 1, 1, 0.5, N[n], N[n], N[n]);
				for(int m=0; m<M; m++){
					if(dim==2)
						fprintf(pFile,"(%f, %f)\n", x[m], y[m]); 
					if(dim==3)
						fprintf(pFile,"(%f, %f, %f)\n", x[m], y[m], z[m]); 
				}
				fclose(pFile);
#endif
			}
		}
	}
	return EXIT_SUCCESS;

}
