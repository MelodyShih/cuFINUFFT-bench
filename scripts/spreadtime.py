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
	output=subprocess.call(['nvidia-smi'])
	if output is not 0:
		return False
	else:
		return True

def get_hostname():
	return socket.gethostname()

def main(OUTPUT, has_gpu=False, hostname=None, resultdir=None):
	d = datetime.datetime.today()
	date = d.strftime('%m%d')
	finufft_filename=(resultdir+hostname+'_finufft_spread_'+date)
	cufinufft_filename=(resultdir+hostname+'_cufinufft_spread_'+date)
	cunfft_filename=(resultdir+hostname+'_cunfft_spread_'+date)

	nupts_distr=1
	dim=2
	reps=5
	density_totry = [1] #0.1, 1, 10
	tol_totry     = [1e-7] #1e-14, 1e-10, 1e-6 , 1e-2
	N1_totry      = [64, 128, 256, 512, 1024, 2048] #128, ... ,4096 
	if OUTPUT is True:
		np.savez(resultdir+'param_'+date, density=density_totry, tol=tol_totry, n1=N1_totry)

	finufft_spread   = np.zeros([len(density_totry), len(tol_totry), len(N1_totry)])
	if has_gpu is True:
		cufinufft_spread = np.zeros([len(density_totry), len(tol_totry), len(N1_totry)])
		cunfft_spread    = np.zeros([len(density_totry), len(tol_totry), len(N1_totry)])

	for d,density in enumerate(density_totry):
		for t,tol in enumerate(tol_totry):
			for n,N1 in enumerate(N1_totry):
				finufft_tnow = float('Inf')
				cufinufft_tnow = float('Inf')
				cunfft_tnow = float('Inf')
				N2=1
				N3=1
				if(dim>1):
					N2=N1
				if(dim>2):
					N3=N1
				M = int(N1*N2*N3*density)
				print('(N1, N2, N3, M, tol)=(%d, %d, %d, %d, %f)' %(N1, N2, N3, M, tol))
				for nn in range(reps):
					tt = 0.0
					finufft_output=subprocess.check_output(["./finufft",\
						str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
						str(M),str(tol)], cwd="../").decode("utf-8")
					finufft_t = float(find_between(finufft_output, \
						"spread (ier=0):", "s"))
					finufft_tnow = min(finufft_tnow,finufft_t)

					if has_gpu is True:
						cufinufft_output=subprocess.check_output(["./cufinufft",\
							str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
							str(M),str(tol)], cwd="../").decode("utf-8")
						cunfft_output=subprocess.check_output(["./cunfft",\
							str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
							str(M),str(tol)], cwd="../").decode("utf-8")
						cufinufft_t = float(find_between(cufinufft_output, \
							"Spread (2)", "s"))
						cufinufft_tnow = min(cufinufft_tnow,cufinufft_t)
						cunfft_t = float(find_between(cunfft_output, \
							"Spread:", "s"))
						cunfft_tnow = min(cunfft_tnow,cunfft_t)

				finufft_spread[d,t,n]   = finufft_tnow
				if has_gpu is True:
					cufinufft_spread[d,t,n] = cufinufft_tnow
					cunfft_spread[d,t,n]   = cunfft_tnow
	if OUTPUT is True:
		np.save(finufft_filename+'.npy', finufft_spread)
		if has_gpu is True:
			np.save(cufinufft_filename+'.npy', cufinufft_spread)
			np.save(cunfft_filename+'.npy', cunfft_spread)

if __name__== "__main__":
	HAS_GPU = has_gpu()
	HOSTNAME = get_hostname()
	main(OUTPUT=True,has_gpu=HAS_GPU,hostname=HOSTNAME,resultdir='../results/')
