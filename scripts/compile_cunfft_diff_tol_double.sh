#!/bin/zsh
cutoff=$1

# Compile CUNFFT with new cutoff variable
cd ~/CUNFFT/build/  
make distclean  
cmake .. -DCUT_OFF=$cutoff -DCUNFFT_DOUBLE_PRECISION=ON -DMILLI_SEC=ON \
         -DMEASURED_TIMES=ON -DCUDA_CUDA_LIBRARY=/usr/lib64/libcuda.so -DPRINT_CONFIG=OFF \
         -DCOM_FG_PSI=ON 
make 
make install

# Compile the timing code
cd ~/cuFINUFFT-bench
rm cunfft_type1_64
rm cunfft_type2_64
make double

# Go back the original directory
cd scripts
