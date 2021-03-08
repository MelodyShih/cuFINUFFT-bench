CC=gcc
CXX=g++
NVCC=nvcc

NVCCFLAGS=-arch=sm_70 -DGPU
CXXFLAGS=#-DACCURACY

FINUFFT=/mnt/home/yshih/finufft
CUFINUFFT=/mnt/home/yshih/cufinufft
CUNFFT=/mnt/home/yshih/CUNFFT

EIGEN=/mnt/home/yshih/eigen-eigen-323c052e1731

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< \
		   -o $@

cunfft_type1: cunfft_timing_type1.cpp utils.o
	$(NVCC) $^ $(NVCCFLAGS) -DSINGLE\
		-I$(EIGEN)/\
		-I$(CUNFFT)/src/\
		-L$(CUNFFT)/lib/\
		-lcunfft -lcudart -lcufft\
		-o $@

cunfft_type1_64: cunfft_timing_type1_double.cpp utils.o
	$(NVCC) $^ $(NVCCFLAGS)\
		-I$(EIGEN)/\
		-I$(CUNFFT)/src/\
		-L$(CUNFFT)/lib/\
		-lcunfft -lcudart -lcufft\
		-o $@

cufinufft_type1: cufinufft_timing_type1.cpp
	$(NVCC) $^ $(NVCCFLAGS) -DSINGLE\
		-I$(EIGEN)/\
		-I$(CUFINUFFT)/include/\
		-L$(CUFINUFFT)/lib\
		-lcudart -lcufinufft -lcufft\
		-o $@

cufinufft_type1_64: cufinufft_timing_type1_double.cpp
	$(NVCC) $^ $(NVCCFLAGS) \
		-I$(EIGEN)/\
		-I$(CUFINUFFT)/include/\
		-L$(CUFINUFFT)/lib\
		-lcudart -lcufinufft -lcufft\
		-o $@

finufft_type1: finufft_timing_type1.cpp
	$(CXX) $^ -DSINGLE -fopenmp $(CXXFLAGS)\
		-I$(EIGEN)/\
		-I$(FINUFFT)/include \
		-L$(FINUFFT)/lib\
		-lfinufft -lfftw3f -lfftw3f_omp\
		-o $@

finufft_type1_64: finufft_timing_type1_double.cpp
	$(CXX) $^ -fopenmp $(CXXFLAGS)\
		-I$(EIGEN)/\
		-I$(FINUFFT)/include \
		-L$(FINUFFT)/lib\
		-lfinufft -lfftw3 -lfftw3f_omp\
		-o $@

cunfft_type2: cunfft_timing_type2.cpp utils.o
	$(NVCC) $^ $(NVCCFLAGS) -DSINGLE\
		-I$(EIGEN)/\
		-I$(CUNFFT)/src/\
		-L$(CUNFFT)/lib/\
		-lcunfft -lcudart -lcufft\
		-o $@

cunfft_type2_64: cunfft_timing_type2_double.cpp utils.o
	$(NVCC) $^ $(NVCCFLAGS) \
		-I$(EIGEN)/\
		-I$(CUNFFT)/src/\
		-L$(CUNFFT)/lib/\
		-lcunfft -lcudart -lcufft\
		-o $@

cufinufft_type2: cufinufft_timing_type2.cpp
	$(NVCC) $^ $(NVCCFLAGS) -DSINGLE\
		-I$(EIGEN)/\
		-I$(CUFINUFFT)/include/\
		-L$(CUFINUFFT)/lib\
		-lcudart -lcufinufft -lcufft\
		-o $@

cufinufft_type2_64: cufinufft_timing_type2_double.cpp
	$(NVCC) $^ $(NVCCFLAGS) \
		-I$(EIGEN)/\
		-I$(CUFINUFFT)/include/\
		-L$(CUFINUFFT)/lib\
		-lcudart -lcufinufft -lcufft\
		-o $@

finufft_type2: finufft_timing_type2.cpp
	$(CXX) $^ -DSINGLE -fopenmp $(CXXFLAGS)\
		-I$(EIGEN)/\
		-I$(FINUFFT)/include \
		-L$(FINUFFT)/lib\
		-lfinufft -lfftw3f -lfftw3f_omp\
		-o $@

finufft_type2_64: finufft_timing_type2_double.cpp
	$(CXX) $^ -fopenmp $(CXXFLAGS)\
		-I$(EIGEN)/\
		-I$(FINUFFT)/include \
		-L$(FINUFFT)/lib\
		-lfinufft -lfftw3 -lfftw3f_omp\
		-o $@

finufft_write_truesol: finufft_write_truesol.cpp
	$(CXX) $^ -fopenmp $(CXXFLAGS) -DSINGLE\
		-I$(EIGEN)/\
		-I$(FINUFFT)/include \
		-L$(FINUFFT)/lib\
		-lfinufft -lfftw3f -lfftw3f_omp\
		-o $@

finufft_write_truesol_64: finufft_write_truesol_64.cpp
	$(CXX) $^ -fopenmp $(CXXFLAGS)\
		-I$(EIGEN)/\
		-I$(FINUFFT)/include \
		-L$(FINUFFT)/lib\
		-lfinufft -lfftw3 -lfftw3f_omp\
		-o $@

writedata: write_data2files.cpp
	$(CXX) $^ -DSINGLE\
		-I$(EIGEN)/\
		-o $@

double: cufinufft_type1_64 finufft_type1_64\
        cufinufft_type2_64 finufft_type2_64\
        cunfft_type1_64 cunfft_type2_64 finufft_write_truesol_64

single: cunfft_type1 cufinufft_type1 finufft_type1\
        cunfft_type2 cufinufft_type2 finufft_type2\
        finufft_write_truesol
clean: 
	rm cunfft_type1 
	rm cufinufft_type1
	rm finufft_type1
	rm cunfft_type2 
	rm cufinufft_type2
	rm finufft_type2
	rm cunfft_type1_64 
	rm cufinufft_type1_64
	rm finufft_type1_64
	rm cunfft_type2_64 
	rm cufinufft_type2_64
	rm finufft_type2_64
	rm finufft_write_truesol
	rm finufft_write_truesol_64
