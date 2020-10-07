import os
import subprocess
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
	d = datetime.datetime.today()
	date = d.strftime('%m%d')

	nupts_distr=3
	dim=2

	finufft_filename=(resultdir+hostname+'_finufft_2d2_'+str(dim)+'d_t'+str(nupts_distr)+\
		'_single_'+date)
	cufinufft_m1_filename=(resultdir+hostname+'_cufinufft_m1_2d2_'+str(dim)+'d_t'+str(nupts_distr)+\
		'_single_'+date)
	cunfft_filename=(resultdir+hostname+'_cunfft_2d2_'+str(dim)+'d_t'+str(nupts_distr)+'_single_'\
		+date)

	reps=5
	tol_totry     = [1e-6] #1e-14, 1e-10, 1e-6 , 1e-2
	density_totry = [0.1, 1, 10] #0.1, 1, 10
	N1_totry      = [64, 128, 256, 512, 1024, 2048] #128, ... ,4096 
	if OUTPUT is True:
		np.savez(resultdir+'param_2d1'+'_'+date, density=density_totry, tol=tol_totry, 
			n1=N1_totry)

	finufft = np.zeros([len(density_totry), len(tol_totry), len(N1_totry), 2])
	if has_gpu is True:
		cufinufft_m1 = np.zeros([len(density_totry), len(tol_totry), len(N1_totry), 2])
		cunfft       = np.zeros([len(density_totry), len(tol_totry), len(N1_totry), 2])

	print('code den N1 N2 N3 M tol exec total')
	for t,tol in enumerate(tol_totry):
		if has_gpu is True:
			FNULL = open(os.devnull, 'w')
			cutoff=(int(np.ceil((-np.ceil(np.log10(tol/10))-2)/2.0)))
			subprocess.call(["sh","compile_cunfft_diff_tol.sh", str(cutoff)], stdout=FNULL,stderr=subprocess.STDOUT)
		for d,density in enumerate(density_totry):
			for n,N1 in enumerate(N1_totry):
				finufft_tnow      = [float('Inf'), float('Inf')]
				cufinufft_m1_tnow = [float('Inf'), float('Inf')]
				cunfft_tnow = float('Inf')
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
						finufft_output=subprocess.check_output(["./finufft_type2",\
							str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
							str(M),str(tol)], cwd="../").decode("utf-8")
						if finufft_output is not None:
							finufft_t = float(find_between(finufft_output, \
								"exec:", "s"))
							finufft_tnow[0] = min(finufft_tnow[0],finufft_t)
							finufft_t = float(find_between(finufft_output, \
								"Totaltime:", "s"))
							finufft_tnow[1] = min(finufft_tnow[1],finufft_t)
						else:
							finufft_t = [float('Inf'), float('Inf)')]
							finufft_tnow[0] = min(finufft_tnow[0],finufft_t[0])
							finufft_tnow[1] = min(finufft_tnow[1],finufft_t[1])
					if has_gpu is True:
						try:
							cufinufft_m1_output=subprocess.check_output(["./cufinufft_type2",\
								str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
								str(M),str(tol),str(1)], cwd="../").decode("utf-8")
						except subprocess.CalledProcessError as e:
							cufinufft_m1_output=None
						try:
							cunfft_output=subprocess.check_output(["./cunfft_type2",\
								str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
								str(M),str(tol)], cwd="../").decode("utf-8")
						except subprocess.CalledProcessError as e:
							cunfft_output=None
						if cufinufft_m1_output is not None:
							cufinufft_m1_t = float(find_between(cufinufft_m1_output, \
								"exec:", "s"))
							cufinufft_m1_tnow[0] = min(cufinufft_m1_tnow[0],cufinufft_m1_t)
							cufinufft_m1_t = float(find_between(cufinufft_m1_output, \
								"Totaltime:", "s"))
							cufinufft_m1_tnow[1] = min(cufinufft_m1_tnow[1],cufinufft_m1_t)
						else:
							cufinufft_m1_t = [float('Inf'), float('Inf')]
							cufinufft_m1_tnow[0] = min(cufinufft_m1_tnow[0],cufinufft_m1_t[0])
							cufinufft_m1_tnow[1] = min(cufinufft_m1_tnow[1],cufinufft_m1_t[1])

						if cunfft_output is not None:
							cunfft_t = float(find_between(cunfft_output, \
								"Totaltime:", "s"))
							cunfft_tnow = min(cunfft_tnow,cunfft_t)
						else:
							cunfft_t = float('Inf')
							cunfft_tnow = min(cunfft_tnow,cunfft_t)
				if has_gpu is not True:
					finufft[d,t,n,0]   = finufft_tnow[0]
					finufft[d,t,n,1]   = finufft_tnow[1]
					print('%d %3.1e %4d %4d %4d %10d %5.3e %5.3e %5.3e' \
						%(0, density, N1, N2, N3, M, tol, finufft_tnow[0], finufft_tnow[1]))
				if has_gpu is True:
					cufinufft_m1[d,t,n,0] = cufinufft_m1_tnow[0]
					cufinufft_m1[d,t,n,1] = cufinufft_m1_tnow[1]
					cunfft[d,t,n,0]= cunfft_tnow
					cunfft[d,t,n,1]= cunfft_tnow

					print('%d %3.1e %4d %4d %4d %10d %5.3e %5.3e %5.3e' \
						%(1, density, N1, N2, N3, M, tol, cufinufft_m1_tnow[0], cufinufft_m1_tnow[1]))
					print('%d %3.1e %4d %4d %4d %10d %5.3e %5.3e %5.3e' \
						%(3, density, N1, N2, N3, M, tol, cunfft_tnow, cunfft_tnow))

	if OUTPUT is True:
		if has_gpu is True:
			np.save(cufinufft_m1_filename+'.npy', cufinufft_m1)
			np.save(cunfft_filename+'.npy', cunfft)
		else:
			np.save(finufft_filename+'.npy', finufft)

if __name__== "__main__":
	HAS_GPU = has_gpu()
	HOSTNAME = get_hostname()
	main(OUTPUT=True,has_gpu=HAS_GPU,hostname=HOSTNAME,resultdir='../results/')
