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

cunfft: cunfft_timing.cpp utils.o
	$(CXX) $^ -DGPU\
		-I$(EIGEN)/\
		-I$(CUNFFT)/src/\
		-L$(CUNFFT)/lib/\
		-lcunfft -lcudart\
		-o $@

cufinufft: cufinufft_timing.cpp
	$(NVCC) $^ -DGPU -DSINGLE $(NVCCFLAGS) \
		-I$(EIGEN)/\
		-I$(CUFINUFFT)/src/\
		-L$(CUFINUFFT)/lib\
		-lcudart -lcufinufftf\
		-o $@

finufft: finufft_timing.cpp
	$(CXX) $^ -DSINGLE -fopenmp\
		-I$(EIGEN)/\
		-I$(FINUFFT)/src \
		-L$(FINUFFT)/lib\
		-lfinufftf -lfftw3f -lfftw3f_omp\
		-o $@

all: cunfft cufinufft finufft
clean: 
	rm cunfft 
	rm cufinufft
	rm finufft
