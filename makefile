CC=gcc
CXX=g++
NVCC=nvcc

NVCCFLAGS=-arch=sm_70 --default-stream per-thread
CXXFLAGS=

FINUFFT=/mnt/home/yshih/finufft
CUFINUFFT=/mnt/home/yshih/GPUnufftSpreader
CUNFFT=/mnt/home/yshih/CUNFFT

EIGEN=/mnt/home/yshih/eigen-eigen-323c052e1731

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

cunfft_type1: cunfft_timing_type1.cpp utils.o
	$(CXX) $^ -DGPU\
		-I$(EIGEN)/\
		-I$(CUNFFT)/src/\
		-L$(CUNFFT)/lib/\
		-lcunfft -lcudart\
		-o $@

cufinufft_type1: cufinufft_timing_type1.cpp
	$(NVCC) $^ -DGPU -DSINGLE $(NVCCFLAGS) \
		-I$(EIGEN)/\
		-I$(CUFINUFFT)/src/\
		-L$(CUFINUFFT)/lib\
		-lcudart -lcufinufftf\
		-o $@

finufft_type1: finufft_timing_type1.cpp
	$(CXX) $^ -DSINGLE -fopenmp\
		-I$(EIGEN)/\
		-I$(FINUFFT)/src \
		-L$(FINUFFT)/lib\
		-lfinufftf -lfftw3f -lfftw3f_omp\
		-o $@

cunfft_type2: cunfft_timing_type2.cpp utils.o
	$(CXX) $^ -DGPU\
		-I$(EIGEN)/\
		-I$(CUNFFT)/src/\
		-L$(CUNFFT)/lib/\
		-lcunfft -lcudart\
		-o $@

cufinufft_type2: cufinufft_timing_type2.cpp
	$(NVCC) $^ -DGPU -DSINGLE $(NVCCFLAGS) \
		-I$(EIGEN)/\
		-I$(CUFINUFFT)/src/\
		-L$(CUFINUFFT)/lib\
		-lcudart -lcufinufftf\
		-o $@

finufft_type2: finufft_timing_type2.cpp
	$(CXX) $^ -DSINGLE -fopenmp\
		-I$(EIGEN)/\
		-I$(FINUFFT)/src \
		-L$(FINUFFT)/lib\
		-lfinufftf -lfftw3f -lfftw3f_omp\
		-o $@

all: cunfft_type1 cufinufft_type1 finufft_type1 cunfft_type2 cufinufft_type2 finufft_type2
clean: 
	rm cunfft_type1 
	rm cufinufft_type1
	rm finufft_type1
	rm cunfft_type2 
	rm cufinufft_type2
	rm finufft_type2
