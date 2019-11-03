CC=gcc
CXX=g++
NVCC=nvcc

NVCCFLAGS=-arch=sm_70 --default-stream per-thread
CXXFLAGS=

FINUFFT=/mnt/home/yshih/finufft
CUFINUFFT=/mnt/home/yshih/GPUnufftSpreader
CUNFFT=/mnt/home/yshih/CUNFFT
#INC=-I/mnt/home/yshih/CUNFFT/src/ \
	-I/mnt/home/yshih/GPUnufftSpreader/src/ \
	-I/mnt/home/yshih/finufft/src/
#LIBS_PATH=-L/mnt/home/yshih/CUNFFT/lib/ \
		  -L/mnt/home/yshih/GPUnufftSpreader/lib \
		  -L/mnt/home/yshih/finufft/lib
#LIBS=-lcunfft -lcudart -lcufinufftf -lfinufftf -lfftw3_omp

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@
#%.o: %.cu
#	$(NVCC) -c $(NVCCFLAGS) $(INC) $< -o $@


cunfft: cunfft_timing.cpp utils.o
	$(CXX) $^ -DGPU\
		-I$(CUNFFT)/src/\
		-L$(CUNFFT)/lib/\
		-lcunfft -lcudart\
		-o $@

cufinufft: cufinufft_timing.cpp
	$(NVCC) $^ -DGPU -DSINGLE $(NVCCFLAGS) \
		-I$(CUFINUFFT)/src/\
		-L$(CUFINUFFT)/lib\
		-lcudart -lcufinufftf\
		-o $@

finufft: finufft_timing.cpp
	$(CXX) $^ -DSINGLE -fopenmp\
		-I$(FINUFFT)/src \
		-L$(FINUFFT)/lib\
		-lfinufftf -lfftw3f -lfftw3f_omp\
		-o $@

all: cunfft cufinufft finufft
clean: 
	rm cunfft 
	rm cufinufft
	rm finufft
