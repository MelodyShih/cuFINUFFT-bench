# cuFINUFFT-bench
Benchmarking for cuFINUFFT library

### CPU code
- FINUFFT: https://github.com/flatironinstitute/finufft

### GPU code
- CUNFFT: https://github.com/sukunis/CUNFFT
- cuFINUFFT: https://github.com/MelodyShih/cufinufft

## Installing CUNFFT
```
cd build/  
make distclean  
cmake .. -DCUT_OFF=4 -DCUNFFT_DOUBLE_PRECISION=OFF -DMILLI_SEC=ON \
         -DMEASURED_TIMES=ON -DCUDA_CUDA_LIBRARY=/loc/to/libcuda.so -DPRINT_CONFIG=OFF  
make 
make install
```
 
cmake FLAG setting
* -DCUT_OFF (the spreading width is `FILTER_SIZE = (2*CUT_OFF+2)`, defined in `cunfft_typedefs.h`)
* -DCUNFFT_DOUBLE_PRECISION
* -DMEASURED_TIMES
* -DCUDA_CUDA_LIBRARY
* -DMILLI_SEC

## Installing FINUFFT
```
make PREC=SINGLE lib
```
## Installing cuFINUFFT
```
make PREC=SINGLE lib
```

## Details for different libraries
### Type 1 transform: nonuniform pts to uniform pts
1. FINUFFT/cuFINUFFT: 
  <img src="http://latex.codecogs.com/gif.latex?f_k = \sum_{j=1}^M c_j e^{ikx_j},~x_j \in [-\pi, \pi]^d" border="0"/>
  
  Input:
  
  * `x`: a size `M` double (float) array of x coordinates of `M` source points
  * `y`: a size `M` double (float) array of y coordinates of `M` source points
  * `z`: a size `M` double (float) array of z coordinates of `M` source points, or `NULL`
  * `c`: a size `M` complex double (float) array of `M` strengths
  * `eps`: relative l2 tolerance, i.e. <img src="http://latex.codecogs.com/gif.latex?\frac{\|f_k - f\|_2}{\|f\|_2}\leq \epsilon" border="0"/>. The spreading width is `-log_10 (eps/10)`
  
  Output:
  
  * `f_k`: a size `N1xN2x...xNd` complex double (float) array of `N1xN2x...xNd` output modes. The modes are ordered from `-N/2` to `N/2-1` in each dimension.
2. CUNFFT 
  <img src="http://latex.codecogs.com/gif.latex?f_k = \sum_{j=1}^M c_j e^{2{\pi} ik x_j},~x_j \in [-0.5, 0.5]^d" border="0"/>
  
  Input:
  
  * `x`: a size `dM` double (float) array of coordinates of `M` source points, the `t`-th coordinate of the `j`-th source point is the `d*j+t` entry of `x`.
  * `c`: a complex double (float) array of `M` strengths
  
  Output:
  
  * `f_k` = a size `N1xN2x...xNd`complex double (float) array of `N1xN2x...xNd` modes. The modes are ordered from `-N/2` to `N/2-1` in each dimension.

 
## Usage
- [Step 1] Install all the libraries
- [Step 2] Add shared libraries path to LD_LIBRARY_PATH
- [Step 3] Execute python scripts
