CC=gcc
CXX=g++
NVCC=nvcc

NVCCFLAGS=-arch=sm_70 --default-stream per-thread
CXXFLAGS=
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
		-I/mnt/home/yshih/CUNFFT/src/\
		-L/mnt/home/yshih/CUNFFT/lib/\
		-lcunfft -lcudart\
		-o $@

cufinufft: cufinufft_timing.cpp
	$(NVCC) $^ -DGPU -DSINGLE $(NVCCFLAGS) \
		-I/mnt/home/yshih/GPUnufftSpreader/src/\
		-L/mnt/home/yshih/GPUnufftSpreader/lib\
		-lcudart -lcufinufftf\
		-o $@

finufft: finufft_timing.cpp
	$(CXX) $^ -DSINGLE \
		-I/mnt/home/yshih/finufft/src \
		-L/mnt/home/yshih/finufft/lib\
		-lfinufftf -lfftw3_omp\
		-o $@

all: cunfft cufinufft finufft
clean: 
	rm cunfft 
	rm cufinufft
	rm finufft
