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

	nupts_distr=2
	dim=2

	finufft_filename=(resultdir+hostname+'_finufft_spread_'+str(dim)+'d_t'+str(nupts_distr)+\
		'_double_'+date)
	cufinufft_m1_filename=(resultdir+hostname+'_cufinufft_m1_spread_'+str(dim)+'d_t'+str(nupts_distr)+\
		'_double_'+date)
	cufinufft_m2_filename=(resultdir+hostname+'_cufinufft_m2_spread_'+str(dim)+'d_t'+str(nupts_distr)+\
		'_double'+date)
	cunfft_filename=(resultdir+hostname+'_cunfft_spread_'+str(dim)+'d_t'+str(nupts_distr)+'_double_'\
		+date)

	reps=5
	tol_totry     = [1e-3, 1e-7] #1e-14, 1e-10, 1e-6 , 1e-2
	density_totry = [0.1, 1, 10] #0.1, 1, 10
	N1_totry      = [64, 128, 256, 512, 1024, 2048] #128, ... ,4096 
	if OUTPUT is True:
		np.savez(resultdir+'param_spread'+'_'+date, density=density_totry, tol=tol_totry, 
			n1=N1_totry)

	finufft_spread   = np.zeros([len(density_totry), len(tol_totry), len(N1_totry)])
	if has_gpu is True:
		cufinufft_m1_spread = np.zeros([len(density_totry), len(tol_totry), len(N1_totry)])
		cufinufft_m2_spread = np.zeros([len(density_totry), len(tol_totry), len(N1_totry)])
		cunfft_spread    = np.zeros([len(density_totry), len(tol_totry), len(N1_totry)])

	for t,tol in enumerate(tol_totry):
		if has_gpu is True:
			cutoff=int((-np.log10(tol/10)-2)/2)
			subprocess.call(["sh","compile_cunfft_diff_tol.sh", str(cutoff)])
		for d,density in enumerate(density_totry):
			for n,N1 in enumerate(N1_totry):
				finufft_tnow = float('Inf')
				cufinufft_m1_tnow = float('Inf')
				cufinufft_m2_tnow = float('Inf')
				cunfft_tnow = float('Inf')
				N2=1
				N3=1
				if(dim>1):
					N2=N1
				if(dim>2):
					N3=N1
				M = int(N1*N2*N3*pow(2,dim)*density)
				print('(den, N1, N2, N3, M, tol)=(%f, %d, %d, %d, %d, %e)' %(density, N1, N2, N3, M, tol))
				for nn in range(reps):
					tt = 0.0
					if has_gpu is not True:
						finufft_output=subprocess.check_output(["./finufft_type1",\
							str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
							str(M),str(tol)], cwd="../").decode("utf-8")
						finufft_t = float(find_between(finufft_output, \
							"spread (ier=0):", "s"))
						finufft_tnow = min(finufft_tnow,finufft_t)

					if has_gpu is True:
						try:
							cufinufft_m1_output=subprocess.check_output(["./cufinufft_type1",\
								str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
								str(M),str(tol),str(1)], cwd="../").decode("utf-8")
						except subprocess.CalledProcessError as e:
							cufinufft_m1_output=None
						try:
							cufinufft_m2_output=subprocess.check_output(["./cufinufft_type1",\
								str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
								str(M),str(tol),str(2)], cwd="../").decode("utf-8")
						except subprocess.CalledProcessError as e:
							cufinufft_m2_output=None
						try:
							cunfft_output=subprocess.check_output(["./cunfft_type1",\
								str(nupts_distr),str(dim),str(N1),str(N2),str(N3),\
								str(M),str(tol)], cwd="../").decode("utf-8")
						except subprocess.CalledProcessError as e:
							cunfft_output=None
						if cufinufft_m1_output is not None:
							cufinufft_m1_t = float(find_between(cufinufft_m1_output, \
								"Spread (1)", "s")) + \
								float(find_between(cufinufft_m1_output, \
								"Setup Subprob properties", "s"))
							cufinufft_m1_tnow = min(cufinufft_m1_tnow,cufinufft_m1_t)
						else:
							cufinufft_m1_t = float('Inf')
							cufinufft_m1_tnow = min(cufinufft_m1_tnow,cufinufft_m1_t)

						if cufinufft_m2_output is not None:
							cufinufft_m2_t = float(find_between(cufinufft_m2_output, \
								"Spread (2)", "s")) + \
								float(find_between(cufinufft_m2_output, \
                            	"Setup Subprob properties", "s"))
							cufinufft_m2_tnow = min(cufinufft_m2_tnow,cufinufft_m2_t)
						else:
							cufinufft_m2_t = float('Inf')
							cufinufft_m2_tnow = min(cufinufft_m2_tnow,cufinufft_m2_t)

						if cunfft_output is not None:
							cunfft_t = float(find_between(cunfft_output, \
								"Spread:", "s"))
						else:
							cunfft_t = float('Inf')
						cunfft_tnow = min(cunfft_tnow,cunfft_t)

				if has_gpu is not True:
					finufft_spread[d,t,n]   = finufft_tnow
				if has_gpu is True:
					cufinufft_m1_spread[d,t,n] = cufinufft_m1_tnow
					cufinufft_m2_spread[d,t,n] = cufinufft_m2_tnow
					cunfft_spread[d,t,n]   = cunfft_tnow
	if OUTPUT is True:
		if has_gpu is True:
			np.save(cufinufft_m1_filename+'.npy', cufinufft_m1_spread)
			np.save(cufinufft_m2_filename+'.npy', cufinufft_m2_spread)
			np.save(cunfft_filename+'.npy', cunfft_spread)
		else:
			np.save(finufft_filename+'.npy', finufft_spread)

if __name__== "__main__":
	HAS_GPU = has_gpu()
	HOSTNAME = get_hostname()
	main(OUTPUT=True,has_gpu=HAS_GPU,hostname=HOSTNAME,resultdir='../results/')
