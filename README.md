# cuFINUFFT-bench
Benchmarking for cuFINUFFT library

### CPU code
- FINUFFT: https://github.com/flatironinstitute/finufft

### GPU code
- CUNFFT: https://github.com/sukunis/CUNFFT
- cuFINUFFT: https://github.com/flatironinstitute/cufinufft
- gpuNUFFT: https://github.com/andyschwarzl/gpuNUFFT

## Installing cuFINUFFT
```
module load cuda/10.0.130_410.48
module load gcc/7.4.0
make lib
```
## Installing FINUFFT
```
module load cuda/10.0.130_410.48
module load gcc/7.4.0
make lib
```
## Installing CUNFFT
```
module load cuda/10.0.130_410.48
module load gcc/7.4.0
cd CUNFFT
cd build/  
make distclean  
cmake .. -DCUT_OFF=3 -DCUNFFT_DOUBLE_PRECISION=OFF -DMILLI_SEC=ON \
         -DMEASURED_TIMES=ON -DCUDA_CUDA_LIBRARY=/usr/lib64/libcuda.so -DPRINT_CONFIG=OFF  
make 
make install
```
cmake FLAG setting
* -DCUT_OFF (the spreading width is `FILTER_SIZE = (2*CUT_OFF+2)`, defined in `cunfft_typedefs.h`)
* -DCUNFFT_DOUBLE_PRECISION
* -DMEASURED_TIMES
* -DCUDA_CUDA_LIBRARY
* -DMILLI_SEC

## Installing gpuNUFFT
Note: gpuNUFFT uses single precision by default.
```
module load cuda/10.0.130_410.48
module load gcc/7.4.0
module load matlab
cd gpuNUFFT
cd CUDA
mkdir -p build
cd build
cmake .. -DMATLAB_ROOT_DIR=/cm/shared/sw/pkg/vendor/matlab/R2020a
make
```
## Usage
- [Step 1] Install all the libraries
- [Step 2] Add shared libraries path to LD_LIBRARY_PATH

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/finufft/lib:~/CUNFFT/lib:~/cuFINUFFT/lib
```
- [Step 3] Execute python scripts
```
python nuffttype1.py
```

## Details for different libraries
### Type 1 transform: nonuniform pts to uniform pts
1. FINUFFT/cuFINUFFT: 
  <img src="https://latex.codecogs.com/gif.latex?f_{k_1,k_2,k_3}&space;=&space;\sum_{j=1}^M&space;c_j&space;e^{2\pi&space;i(k_1x_j&plus;k_2y_j&plus;k_3z_j)},~x_j,y_j,z_j&space;\in&space;[-\pi,&space;\pi)" title="f_{k_1,k_2,k_3} = \sum_{j=1}^M c_j e^{2\pi i(k_1x_j+k_2y_j+k_3z_j)},~x_j,y_j,z_j \in [-\pi, \pi]" />
  
  Input:
  * `x`: a size `M` float array of x coordinates of `M` source points
  * `y`: a size `M` float array of y coordinates of `M` source points
  * `z`: a size `M` float array of z coordinates of `M` source points, or `NULL`
  * `c`: a size `M` complex float array of `M` strengths
  * `eps`: relative l2 tolerance, i.e. <img src="https://latex.codecogs.com/gif.latex?\frac{\|\boldsymbol{f_k}&space;-&space;\boldsymbol{f}\|_2}{\|\boldsymbol{f}\|_2}\leq&space;\epsilon" title="\frac{\|\boldsymbol{f_k} - \boldsymbol{f}\|_2}{\|\boldsymbol{f}\|_2}\leq \epsilon" />. The spreading width is `-log_10 (eps/10)`
  
  Output:
  
  * `f_k`: a size `N1xN2x...xNd` complex float array of `N1xN2x...xNd` output modes. The modes are ordered from `-N/2` to `N/2-1` in each dimension and are ordered first in dimension `x`, then in `y` and last in `z`, i.e. in 3D, `f_k[k1+k2*N1+k3*N1*N2]` approximates <img src="https://latex.codecogs.com/gif.latex?\sum_{j=1}^{M}c_je^{i\left(k_1&plus;N_1/2)x_j&space;&plus;&space;(k_2&plus;N_2/2)y_j&space;&plus;&space;(k_3&plus;N_3/2)z_j\right)}" title="\sum_{j=1}^{M}c_je^{i\left(k_1+N_1/2)x_j + (k_2+N_2/2)y_j + (k_3+N_3/2)z_j\right)}" />.

2. CUNFFT 
  <img src="https://latex.codecogs.com/gif.latex?f_{k_1,k_2,k_3}&space;=&space;\sum_{j=1}^M&space;c_j&space;e^{2\pi&space;i(k_1x_j&plus;k_2y_j&plus;k_3z_j)},~x_j,y_j,z_j&space;\in&space;[-0.5,&space;0.5)" title="f_{k_1,k_2,k_3} = \sum_{j=1}^M c_j e^{2\pi i(k_1x_j+k_2y_j+k_3z_j)},~x_j,y_j,z_j \in [-0.5, 0.5]" />
  
  Input:
  
  * `x`: a size `dM` float array of coordinates of `M` source points, the `t`-th coordinate of the `j`-th source point is the `d*j+t` entry of `x`.
  * `c`: a complex float array of `M` strengths
  
  Output:
  
  * `f_k` = a size `N1xN2x...xNd`complex float array of `N1xN2x...xNd` modes. The modes are ordered from `-N/2` to `N/2-1` in each dimension. Modes are ordered first in dimension `z`, then in `y` and last in `x`, i.e. in 3D, `f_k[k_3+k_2*N3+k_1*N3*N2]` is the Fourier coefficient approximates <img src="https://latex.codecogs.com/gif.latex?\sum_{j=1}^{M}c_je^{2\pi&space;i\left((k_1&plus;N_1/2)x_j&space;&plus;&space;(k_2&plus;N_2/2)y_j&space;&plus;&space;(k_3&plus;N_3/2)z_j\right)}" title="\sum_{j=0}^{M-1}c_je^{2\pi i\left((k_1+N_1/2)x_j + (k_2+N_2/2)y_j + (k_3+N_3/2)z_j\right)}" />.
  
3. gpuNUFFT
<img src="http://latex.codecogs.com/svg.latex?f_{k_1,k_2,k_3}&space;=&space;\frac{1}{\prod_{i=1}^d&space;\sqrt{2N_i}}\sum_{j=1}^M&space;c_j&space;e^{2\pi&space;i(k_1x_j&plus;k_2y_j&plus;k_3z_j)},~x_j,y_j,z_j&space;\in&space;[-0.5,&space;0.5]" title="http://latex.codecogs.com/svg.latex?f_{k_1,k_2,k_3} = \frac{1}{\prod_{i=1}^d \sqrt{2N_i}}\sum_{j=1}^M c_j e^{2\pi i(k_1x_j+k_2y_j+k_3z_j)},~x_j,y_j,z_j \in [-0.5, 0.5]" />

Input:

* `k`: a size `d` by `M` float matrix of coordinates of `M` source points, the `t`-th coordinate of the `j`-th source point is the `(t,j)` entry of `k`.
* `c`: a complex float array of `M` strengths.

Output:

* `f_k` = a size `N1` by `N2` (by `N3`) complex float d-dimensional array of `N1xN2x...xNd` modes. The modes are ordered from `-N/2` to `N/2-1` in each dimension. `f_k(k1, k2, k3)` is the Fourier coefficient approximates <img src="https://latex.codecogs.com/gif.latex?&space;&space;\frac{1}{\prod_{i=1}^3&space;\sqrt{2N_i}}\sum_{j=1}^{M}c_je^{2\pi&space;i\left((k_1&plus;N_1/2)x_j&space;&plus;&space;(k_2&plus;N_2/2)y_j&space;&plus;&space;(k_3&plus;N_3/2)z_j\right)}" title="\frac{1}{\prod_{i=1}^d\sum_{j=1}^{M}c_je^{2\pi i\left((k_1+N_1/2)x_j + (k_2+N_2/2)y_j + (k_3+N_3/2)z_j\right)}" />.
