# cuFINUFFT-bench
Benchmarking for cuFINUFFT library

### CPU code
- FINUFFT: https://github.com/flatironinstitute/finufft

### GPU code
- CUNFFT: https://github.com/sukunis/CUNFFT
- CUFINUFFT: https://github.com/MelodyShih/cufinufft

## Installing CUNFFT
cmake FLAG setting
-DCUT_OFF
-DCUNFFT_DOUBLE_PRECISION
-DMEASURED_TIMES
-DCUDA_CUDA_LIBRARY
-DMILLI_SEC

## Usage
- [Step 1] Install all the libraries
- [Step 2] Add shared libraries path to LD_LIBRARY_PATH
- [Step 3] Execute python scripts
