#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cub/cub.cuh>

using namespace std;

/*
	Compile with nvcc thrust_vs_cub.cu -I/loc/to/cub
*/
int main(int argc, char* argv[]){
	int N = 10;
	int lib = 1;
	int reps = 10;
	if (argc<4) {
		fprintf(stderr,"Usage: ./a.out [numberofitem2scan [lib [reps]]]\n");
		return 1;
	}  

	if(argc>1){
		sscanf(argv[1],"%d",&N);
	}
	if(argc>2){
		sscanf(argv[2],"%d",&lib);
	}
	if(argc>3){
		sscanf(argv[3],"%d",&reps);
	}

	int *h_arr2scan  = (int*) malloc(N*sizeof(int));
	int *h_arrscanned = (int*) malloc(N*sizeof(int));
	int *d_arr2scan;
	int *d_arrscanned;

	cudaEvent_t start, stop;
	float milliseconds = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc(&d_arr2scan      ,N*sizeof(int));
	cudaMalloc(&d_arrscanned    ,N*sizeof(int));

	for(int i=0; i<N; i++){
		h_arr2scan[i] = rand()%10;
		//cout << h_arr2scan[i] << " ";
	}
	//cout << endl;
	cudaMemcpy(d_arr2scan    , h_arr2scan, N*sizeof(int), cudaMemcpyHostToDevice);

	if (lib == 1){
		// using thrust
		cudaEventRecord(start);
		for(int i=0; i<reps; i++){
			thrust::device_ptr<int> d_ptr(d_arr2scan);
			thrust::device_ptr<int> d_result(d_arrscanned);
			thrust::exclusive_scan(d_ptr, d_ptr + N, d_result);
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] Thrust exclusize scan \t\t %.3g s\n", milliseconds/1000/reps);

#if 0
		cudaMemcpy(h_arrscanned, d_arrscanned, N*sizeof(int), cudaMemcpyDeviceToHost);
		for(int i=0; i<N; i++){
			cout << h_arrscanned[i] << " ";
		}
		cout << endl;
#endif
	}else{
		// using cub
		cudaEventRecord(start);
		for(int i=0; i<reps; i++){
			void     *d_temp_storage = NULL;
			size_t   temp_storage_bytes = 0;
			cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_arr2scan,
					d_arrscanned, N);
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_arr2scan,
					d_arrscanned, N);
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);

		printf("[time  ] CUB exclusize scan \t\t %.3g s\n", milliseconds/1000/reps);
#if 0
		cudaMemcpy(h_arrscanned, d_arrscanned, N*sizeof(int), cudaMemcpyDeviceToHost);
		for(int i=0; i<N; i++){
			cout << h_arrscanned[i] << " ";
		}
		cout << endl;
#endif
	}
}
