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

	nupts_distr = 1

	finufft_filename=(resultdir+hostname+'_finufft_unspread_3d_t'+str(nupts_distr)+\
		'_'+date)
	cufinufft_filename=(resultdir+hostname+'_cufinufft_unspread_3d_t'+str(nupts_distr)+\
		'_'+date)
	cunfft_filename=(resultdir+hostname+'_cunfft_unspread_3d_t'+str(nupts_distr)+'_'\
		+date)

	dim=3
	reps=5
	tol_totry     = [1e-3, 1e-7] #1e-14, 1e-10, 1e-6 , 1e-2
	density_totry = [0.1, 1] #0.1, 1, 10
	N1_totry      = [16, 32, 64, 128, 256] #128, ... ,4096 
	if OUTPUT is True:
		np.savez(resultdir+'param_unspread'+'_'+date, density=density_totry, 
			tol=tol_totry, n1=N1_totry)

	finufft_spread   = np.zeros([len(density_totry), len(tol_totry), len(N1_totry)])
	if has_gpu is True:
		cufinufft_spread = np.zeros([len(density_totry), len(tol_totry), len(N1_totry)])
		cunfft_spread    = np.zeros([len(density_totry), len(tol_totry), len(N1_totry)])

	for t,tol in enumerate(tol_totry):
		if has_gpu is True:
			cutoff=int((-np.log10(tol/10)-2)/2)
			subprocess.call(["sh","compile_cunfft_diff_tol.sh", str(cutoff)])
		for d,density in enumerate(density_totry):
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
				M = int(N1*N2*N3*8*density)
				print('(den, N1, N2, N3, M, tol)=(%f, %d, %d, %d, %d, %e)' %(density, N1, N2, N3, M, tol))
				for nn in range(reps):
					tt = 0.0
					if has_gpu is not True:
						finufft_output=subprocess.check_output(["./finufft_type2",\
							str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
							str(M),str(tol)], cwd="../").decode("utf-8")
						finufft_t = float(find_between(finufft_output, \
							"unspread (ier=0):", "s"))
						finufft_tnow = min(finufft_tnow,finufft_t)

					if has_gpu is True:
						try:
							cufinufft_output=subprocess.check_output(["./cufinufft_type2",\
								str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
								str(M),str(tol)], cwd="../").decode("utf-8")
						except subprocess.CalledProcessError as e:
							cufinufft_output=None
						try:
							cunfft_output=subprocess.check_output(["./cunfft_type2",\
								str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
								str(M),str(tol)], cwd="../").decode("utf-8")
						except subprocess.CalledProcessError as e:
							cunfft_output=None
						if cufinufft_output is not None:
							cufinufft_t = float(find_between(cufinufft_output, \
								"Unspread (1)", "s")) + \
								float(find_between(cufinufft_output, \
                            	"Setup Subprob properties", "s"))
							cufinufft_tnow = min(cufinufft_tnow,cufinufft_t)
						else:
							cufinufft_t = float('Inf')
							cufinufft_tnow = min(cufinufft_tnow,cufinufft_t)
							break
						if cunfft_output is not None:
							cunfft_t = float(find_between(cunfft_output, \
								"Unspread:", "s"))
						else:
							cunfft_t = float('Inf')
						cunfft_tnow = min(cunfft_tnow,cunfft_t)

				if has_gpu is not True:
					finufft_spread[d,t,n]   = finufft_tnow
				if has_gpu is True:
					cufinufft_spread[d,t,n] = cufinufft_tnow
					cunfft_spread[d,t,n]   = cunfft_tnow
	if OUTPUT is True:
		if has_gpu is True:
			np.save(cufinufft_filename+'.npy', cufinufft_spread)
			np.save(cunfft_filename+'.npy', cunfft_spread)
		else:
			np.save(finufft_filename+'.npy', finufft_spread)

if __name__== "__main__":
	HAS_GPU = has_gpu()
	HOSTNAME = get_hostname()
	main(OUTPUT=True,has_gpu=HAS_GPU,hostname=HOSTNAME,resultdir='../results/')
