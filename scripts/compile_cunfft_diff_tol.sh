#!/bin/zsh
cutoff=$1

# Compile CUNFFT with new cutoff variable
cd ~/CUNFFT/build/  
make distclean  
cmake .. -DCUT_OFF=$cutoff -DCUNFFT_DOUBLE_PRECISION=OFF -DMILLI_SEC=ON \
         -DMEASURED_TIMES=ON -DCUDA_CUDA_LIBRARY=/usr/lib64/libcuda.so -DPRINT_CONFIG=OFF  
make 
make install

# Compile the timing code
cd ~/cuFINUFFT-bench
make clean
make all

# Go back the original directory
cd scripts
