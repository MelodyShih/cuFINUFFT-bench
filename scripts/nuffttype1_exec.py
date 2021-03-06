import os
import subprocess
from subprocess import Popen, PIPE
import socket
import datetime
import numpy as np
import re

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

def main(OUTPUT, has_gpu=False, hostname=None, resultdir=None):
	dim=3
	reps=3
	tol_totry     = [1e-2, 1e-5] #1e-14, 1e-10, 1e-6 , 1e-2
	cutoff_totry  = [1, 6] #1e-14, 1e-10, 1e-6 , 1e-2
	nuptsdistr_totry =[2]
	if dim == 2:
		density_totry = [0.1, 1, 10] #0.1, 1, 10
		N1_totry      = [64, 128, 256, 512, 1024, 2048] #128, ... ,4096 
	if dim == 3:
		density_totry = [0.1, 1] #0.1, 1, 10
		N1_totry      = [32, 256] #128, ... ,4096 

	print('pid nupts code den N1 N2 N3 M tol spread fft deconvolve other')
	for nupts_distr in nuptsdistr_totry:
		for t,tol in enumerate(tol_totry):
			if has_gpu is True:
				FNULL = open(os.devnull, 'w')
				cutoff=cutoff_totry[t]
				subprocess.call(["sh","compile_cunfft_diff_tol.sh", str(cutoff)], stdout=FNULL,stderr=subprocess.STDOUT)
			for d,density in enumerate(density_totry):
				for n,N1 in enumerate(N1_totry):
					finufft_tnow      = [float('Inf'), float('Inf'), float('Inf')]
					cufinufft_m1_tnow = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
					cufinufft_m2_tnow = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
					cunfft_tnow       = [float('Inf'), float('Inf'), float('Inf')]
					N2=1
					N3=1
					if(dim>1):
						N2=N1
					if(dim>2):
						N3=N1
					M = int(N1*N2*N3*pow(2,dim)*density)
					for nn in range(reps):
						tt = 0.0
						if has_gpu is not True:
							pipe = Popen(["./finufft_type1",\
								str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
								str(M),str(tol)], stdout=PIPE, cwd="../", stderr=subprocess.STDOUT)
							finufft_output = pipe.communicate()[0]
							finufft_pid = pipe.pid
							#finufft_output=subprocess.check_output(["./finufft_type1",\
							#	str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
							#	str(M),str(tol)], cwd="../", stderr=subprocess.STDOUT).decode("utf-8")
							if finufft_output is not None:
								finufft_t = float(find_between(finufft_output, \
									"spread:", "s"))
								finufft_tnow[0] = min(finufft_tnow[0],finufft_t)
								finufft_t = float(find_between(finufft_output, \
									"FFT:", "s"))
								finufft_tnow[1] = min(finufft_tnow[1],finufft_t)
								finufft_t = float(find_between(finufft_output, \
									"deconvolve:", "s"))
								finufft_tnow[2] = min(finufft_tnow[2],finufft_t)
							else:
								finufft_t = [float('Inf'), float('Inf'), float('Inf')]
								finufft_tnow[0] = min(finufft_tnow[0],finufft_t[0])
								finufft_tnow[1] = min(finufft_tnow[1],finufft_t[1])
								finufft_tnow[2] = min(finufft_tnow[2],finufft_t[2])
						if has_gpu is True:
							try:
								pipe = Popen(["./cufinufft_type1",\
										     str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
										     str(M),str(tol),str(1)]
											,stdout=PIPE,cwd="../")
								cufinufft_m1_output= pipe.communicate()[0]
								cufinufft_m1_pid = pipe.pid
								#cufinufft_m1_output=subprocess.check_output(["./cufinufft_type1",\
								#	str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
								#	str(M),str(tol),str(1)], cwd="../").decode("utf-8")
							except subprocess.CalledProcessError as e:
								cufinufft_m1_output=None
							try:
								pipe = Popen(["./cufinufft_type1",\
										     str(nupts_distr),str(dim),str(N2),str(N2),str(N3),\
										     str(M),str(tol),str(2)]
											,stdout=PIPE,cwd="../")
								cufinufft_m2_output= pipe.communicate()[0]
								cufinufft_m2_pid = pipe.pid
								#cufinufft_m2_output=subprocess.check_output(["./cufinufft_type1",\
								#	str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
								#	str(M),str(tol),str(2)], cwd="../",stderr=subprocess.STDOUT).decode("utf-8")
							except subprocess.CalledProcessError as e:
								cufinufft_m2_output=None
							try:
								pipe = Popen(["./cunfft_type1",\
									         str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
									         str(M),str(tol)],stdout=PIPE, cwd="../")
								cunfft_output= pipe.communicate()[0]
								cunfft_pid = pipe.pid
								#cunfft_output=subprocess.check_output(["./cunfft_type1",\
								#	str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
								#	str(M),str(tol)], cwd="../").decode("utf-8")
							except subprocess.CalledProcessError as e:
								cunfft_output=None
							if cufinufft_m1_output is not None:
								cufinufft_m1_t = float(find_between(cufinufft_m1_output, \
									"Initialize fw", "s"))
								cufinufft_m1_tnow[0] = min(cufinufft_m1_tnow[0],cufinufft_m1_t)
								cufinufft_m1_t = float(find_between(cufinufft_m1_output, \
									"Spread (1)", "s"))
								cufinufft_m1_tnow[1] = min(cufinufft_m1_tnow[1],cufinufft_m1_t)
								cufinufft_m1_t = float(find_between(cufinufft_m1_output, \
									"CUFFT Exec", "s"))
								cufinufft_m1_tnow[2] = min(cufinufft_m1_tnow[2],cufinufft_m1_t)
								cufinufft_m1_t = float(find_between(cufinufft_m1_output, \
									"Deconvolve", "s"))
								cufinufft_m1_tnow[3] = min(cufinufft_m1_tnow[3],cufinufft_m1_t)
							else:
								cufinufft_m1_t = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
								cufinufft_m1_tnow[0] = min(cufinufft_m1_tnow[0],cufinufft_m1_t[0])
								cufinufft_m1_tnow[1] = min(cufinufft_m1_tnow[1],cufinufft_m1_t[1])
								cufinufft_m1_tnow[2] = min(cufinufft_m1_tnow[2],cufinufft_m1_t[2])
								cufinufft_m1_tnow[3] = min(cufinufft_m1_tnow[3],cufinufft_m1_t[3])

							if cufinufft_m2_output is not None:
								cufinufft_m2_t = float(find_between(cufinufft_m2_output, \
									"Initialize fw", "s"))
								cufinufft_m2_tnow[0] = min(cufinufft_m2_tnow[0],cufinufft_m2_t)
								cufinufft_m2_t = float(find_between(cufinufft_m2_output, \
									"Spread (2)", "s"))
								cufinufft_m2_tnow[1] = min(cufinufft_m2_tnow[1],cufinufft_m2_t)
								cufinufft_m2_t = float(find_between(cufinufft_m2_output, \
									"CUFFT Exec", "s"))
								cufinufft_m2_tnow[2] = min(cufinufft_m2_tnow[2],cufinufft_m2_t)
								cufinufft_m2_t = float(find_between(cufinufft_m2_output, \
									"Deconvolve", "s"))
								cufinufft_m2_tnow[3] = min(cufinufft_m2_tnow[3],cufinufft_m2_t)
							else:
								cufinufft_m2_t = [float('Inf'), float('Inf'), float('Inf'), float('Inf')]
								cufinufft_m2_tnow[0] = min(cufinufft_m2_tnow[0],cufinufft_m2_t[0])
								cufinufft_m2_tnow[1] = min(cufinufft_m2_tnow[1],cufinufft_m2_t[1])
								cufinufft_m2_tnow[2] = min(cufinufft_m2_tnow[2],cufinufft_m2_t[2])
								cufinufft_m2_tnow[3] = min(cufinufft_m2_tnow[3],cufinufft_m2_t[3])

							if cunfft_output is not None:
								cunfft_t = float(find_between(cunfft_output, \
									"spread:", "s"))
								cunfft_tnow[0] = min(cunfft_tnow[0],cunfft_t)
								cunfft_t = float(find_between(cunfft_output, \
									"fft:", "s"))
								cunfft_tnow[1] = min(cunfft_tnow[1],cunfft_t)
								cunfft_t = float(find_between(cunfft_output, \
									"deconvolve:", "s"))
								cunfft_tnow[2] = min(cunfft_tnow[2],cunfft_t)
							else:
								cunfft_t = [float('Inf'), float('Inf'), float('Inf')]
								cunfft_tnow[0] = min(cunfft_tnow[0],cunfft_t[0])
								cunfft_tnow[1] = min(cunfft_tnow[1],cunfft_t[1])
								cunfft_tnow[2] = min(cunfft_tnow[2],cunfft_t[2])
					if has_gpu is not True:
						print('%d %d %d %3.1e %4d %4d %4d %10d %5.3e %5.3e %5.3e %5.3e %5.3e' \
							%(finufft_pid, nupts_distr, 0, density, N1, N2, N3, M, tol, \
                              finufft_tnow[0], finufft_tnow[1], finufft_tnow[2], 0))
					if has_gpu is True:
						print('%d %d %d %3.1e %4d %4d %4d %10d %5.3e %5.3e %5.3e %5.3e %5.3e' \
							%(cufinufft_m1_pid, nupts_distr, 1, density, N1, N2, N3, M, tol, cufinufft_m1_tnow[1], \
							  cufinufft_m1_tnow[2], cufinufft_m1_tnow[3], cufinufft_m1_tnow[0]))
						print('%d %d %d %3.1e %4d %4d %4d %10d %5.3e %5.3e %5.3e %5.3e %5.3e' \
							%(cufinufft_m2_pid, nupts_distr, 2, density, N1, N2, N3, M, tol, cufinufft_m2_tnow[1], \
							  cufinufft_m2_tnow[2], cufinufft_m2_tnow[3], cufinufft_m2_tnow[0]))
						print('%d %d %d %3.1e %4d %4d %4d %10d %5.3e %5.3e %5.3e %5.3e %5.3e' \
							%(cunfft_pid, nupts_distr, 3, density, N1, N2, N3, M, tol, cunfft_tnow[0], \
							  cunfft_tnow[1], cunfft_tnow[2], 0))

if __name__== "__main__":
	HAS_GPU = has_gpu()
	HOSTNAME = get_hostname()
	main(OUTPUT=False,has_gpu=HAS_GPU,hostname=HOSTNAME,resultdir='../results/')
