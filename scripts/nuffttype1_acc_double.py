import os
import subprocess
import socket
import datetime
import numpy as np
import re
import sys

def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def has_gpu():
	FNULL = open(os.devnull, 'w')
	output=subprocess.call(['nvidia-smi'],stdout=FNULL, stderr=subprocess.STDOUT)
	if output is not 0:
		return False
	else:
		return True

def get_hostname():
	return socket.gethostname()

def main(OUTPUT, has_gpu=False, hostname=None, dim=2, resultdir=None):
	#dim=2
	reps=3
	tol_totry     = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12] #[1e-7] #1e-14, 1e-10, 1e-6 , 1e-2
	cutoff_totry  = [2, 4, 6, 8, 10, 12] #[3]
	nuptsdistr_totry =[1]
	if dim == 2:
		N1=1000 #128, ... ,4096 
		M=1e7
	if dim == 3:
		N1 = 100 #128, ... ,4096 
		M = 1e7

	print('nupts code N1 N2 N3 M exec total totalgpumem acc')
	for nupts_distr in nuptsdistr_totry:
		for t,tol in enumerate(tol_totry):
			if has_gpu is True:
				FNULL = open(os.devnull, 'w')
				cutoff= cutoff_totry[t]
				subprocess.call(["sh","compile_cunfft_diff_tol_double.sh", str(cutoff)], stdout=FNULL,stderr=subprocess.STDOUT)
			finufft_tnow      = [float('Inf'), float('Inf'), float('Inf')]
			cufinufft_m1_tnow = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
			cufinufft_m2_tnow = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
			cunfft_tnow       = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
			N2=1
			N3=1
			if(dim>1):
				N2=N1
			if(dim>2):
				N3=N1
			for nn in range(reps):
				tt = 0.0
				if has_gpu is not True:
					finufft_output=subprocess.check_output(["./finufft_type1_64",\
						str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
						str(M),str(tol)], cwd="../").decode("utf-8")
					if finufft_output is not None:
						finufft_t = float(find_between(finufft_output, \
							"exec:", "s"))
						finufft_tnow[0] = min(finufft_tnow[0],finufft_t)
						finufft_t = float(find_between(finufft_output, \
							"total:", "s"))
						finufft_tnow[1] = min(finufft_tnow[1],finufft_t)
						finufft_t = float(find_between(finufft_output, \
							"releativeerr:", "\n"))
						finufft_tnow[2] = min(finufft_tnow[2],finufft_t)
					else:
						finufft_t = [float('Inf'), float('Inf)'), float('Inf')]
						finufft_tnow[0] = min(finufft_tnow[0],finufft_t[0])
						finufft_tnow[1] = min(finufft_tnow[1],finufft_t[1])
						finufft_tnow[2] = min(finufft_tnow[2],finufft_t[2])
				if has_gpu is True:
					try:
						cufinufft_m1_output=subprocess.check_output(["./cufinufft_type1_64",\
							str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
							str(M),str(tol),str(1)], cwd="../").decode("utf-8")
					except subprocess.CalledProcessError as e:
						cufinufft_m1_output=None
					try:
						cufinufft_m2_output=subprocess.check_output(["./cufinufft_type2_64",\
							str(nupts_distr),str(dim),str(N2),str(N2),str(N3),\
							str(M),str(tol),str(2)], cwd="../").decode("utf-8")
					except subprocess.CalledProcessError as e:
						cufinufft_m2_output=None
					try:
						cunfft_output=subprocess.check_output(["./cunfft_type1_64",\
							str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
							str(M),str(tol)], cwd="../").decode("utf-8")
					except subprocess.CalledProcessError as e:
						cunfft_output=None
					if cufinufft_m1_output is not None:
						cufinufft_m1_t = float(find_between(cufinufft_m1_output, \
							"exec:", "s"))
						cufinufft_m1_tnow[0] = min(cufinufft_m1_tnow[0],cufinufft_m1_t)
						cufinufft_m1_t = float(find_between(cufinufft_m1_output, \
							"total:", "s"))
						cufinufft_m1_tnow[1] = min(cufinufft_m1_tnow[1],cufinufft_m1_t)
						cufinufft_m1_t = float(find_between(cufinufft_m1_output, \
							"total+gpumem:", "s"))
						cufinufft_m1_tnow[2] = min(cufinufft_m1_tnow[2],cufinufft_m1_t)
						cufinufft_m1_t = float(find_between(cufinufft_m1_output, \
							"releativeerr:", "\n"))
						cufinufft_m1_tnow[3] = min(cufinufft_m1_tnow[3],cufinufft_m1_t)
					else:
						cufinufft_m1_t = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
						cufinufft_m1_tnow[0] = min(cufinufft_m1_tnow[0],cufinufft_m1_t[0])
						cufinufft_m1_tnow[1] = min(cufinufft_m1_tnow[1],cufinufft_m1_t[1])
						cufinufft_m1_tnow[2] = min(cufinufft_m1_tnow[2],cufinufft_m1_t[2])
						cufinufft_m1_tnow[3] = min(cufinufft_m1_tnow[3],cufinufft_m1_t[3])
					if cufinufft_m2_output is not None:
						cufinufft_m2_t = float(find_between(cufinufft_m2_output, \
							"exec:", "s"))
						cufinufft_m2_tnow[0] = min(cufinufft_m2_tnow[0],cufinufft_m2_t)
						cufinufft_m2_t = float(find_between(cufinufft_m2_output, \
							"total:", "s"))
						cufinufft_m2_tnow[2] = min(cufinufft_m2_tnow[2],cufinufft_m2_t)
						cufinufft_m2_t = float(find_between(cufinufft_m2_output, \
							"total+gpumem:", "s"))
						cufinufft_m2_tnow[2] = min(cufinufft_m2_tnow[2],cufinufft_m2_t)
						cufinufft_m2_t = float(find_between(cufinufft_m2_output, \
							"releativeerr:", "\n"))
						cufinufft_m2_tnow[3] = min(cufinufft_m2_tnow[3],cufinufft_m2_t)
					else:
						cufinufft_m2_t = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
						cufinufft_m2_tnow[0] = min(cufinufft_m2_tnow[0],cufinufft_m2_t[0])
						cufinufft_m2_tnow[2] = min(cufinufft_m2_tnow[2],cufinufft_m2_t[2])
						cufinufft_m2_tnow[2] = min(cufinufft_m2_tnow[2],cufinufft_m2_t[2])
						cufinufft_m2_tnow[3] = min(cufinufft_m2_tnow[3],cufinufft_m2_t[3])

					if cunfft_output is not None:
						cunfft_t = float(find_between(cunfft_output, \
							"exec:", "s"))
						cunfft_tnow[0] = min(cunfft_tnow[0],cunfft_t)
						cunfft_tnow[1] = min(cunfft_tnow[1],cunfft_t)
						cunfft_t = float(find_between(cunfft_output, \
							"total+gpumem:", "s"))
						cunfft_tnow[2] = min(cunfft_tnow[2],cunfft_t)
						cunfft_t = float(find_between(cunfft_output, \
							"releativeerr:", "\n"))
						cunfft_tnow[3] = min(cunfft_tnow[3],cunfft_t)
					else:
						cunfft_t = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
						cunfft_tnow[0] = min(cunfft_tnow[0],cunfft_t[0])
						cunfft_tnow[1] = min(cunfft_tnow[1],cunfft_t[1])
						cunfft_tnow[2] = min(cunfft_tnow[2],cunfft_t[2])
						cunfft_tnow[3] = min(cunfft_tnow[3],cunfft_t[3])
			if has_gpu is not True:
				print('%d %d %4d %4d %4d %10d %5.3e %5.3e %5.3e %5.3e' \
					%(nupts_distr, 0, N1, N2, N3, M, \
					  finufft_tnow[0], finufft_tnow[1], finufft_tnow[1], finufft_tnow[2]))
			if has_gpu is True:
				print('%d %d %4d %4d %4d %10d %5.3e %5.3e %5.3e %5.3e' \
					%(nupts_distr, 1, N1, N2, N3, M, cufinufft_m1_tnow[0], \
					  cufinufft_m1_tnow[1], cufinufft_m1_tnow[2], cufinufft_m1_tnow[3]))
				print('%d %d %4d %4d %4d %20d %5.3e %5.3e %5.3e %5.3e' \
					%(nupts_distr, 2, N2, N2, N3, M, cufinufft_m2_tnow[0], \
					  cufinufft_m2_tnow[2], cufinufft_m2_tnow[2], cufinufft_m2_tnow[3]))
				print('%d %d %4d %4d %4d %10d %5.3e %5.3e %5.3e %5.3e' \
					%(nupts_distr, 3, N1, N2, N3, M, cunfft_tnow[0], \
					  cunfft_tnow[1], cunfft_tnow[2], cunfft_tnow[3]))

if __name__== "__main__":
	HAS_GPU = has_gpu()
	HOSTNAME = get_hostname()
	main(OUTPUT=False,has_gpu=HAS_GPU,hostname=HOSTNAME,dim=int(sys.argv[-1]),resultdir='../results/')
