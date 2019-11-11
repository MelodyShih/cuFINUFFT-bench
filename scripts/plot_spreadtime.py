import numpy as np
import datetime

import matplotlib.pyplot as plt
from matplotlib import ticker

def main():
	d = datetime.datetime.today()
	date = d.strftime('%m%d')
	nuptstype = 2
	tol = 1e-7

	result_dir="../results/"
	t_cufinufft = np.load(result_dir+'workergpu31_cufinufft_spread_3d_t'+\
		str(nuptstype)+'_['+str(tol)+']_'+date+'.npy')
	t_cufinufft_2 = np.load(result_dir+'workergpu18_cufinufft_spread_3d_t'+\
		str(nuptstype)+'_['+str(tol)+']_'+'maxsubprob1024_'+date+'.npy')
	t_finufft   = np.load(result_dir+'worker1047_finufft_spread_3d_t'+\
		str(nuptstype)+'_['+str(tol)+']_'+date+'.npy')
	t_cunfft    = np.load(result_dir+'workergpu31_cunfft_spread_3d_t'+\
		str(nuptstype)+'_['+str(tol)+']_'+date+'.npy')

	npzfile = np.load(result_dir+'param_['+str(tol)+']_'+date+'.npz')
	dim = 3

	## Create x label
	xlabel = ''
	for i in range(dim):
		xlabel = xlabel+'N'+str(i+1)
		if i+1 < dim:
			xlabel = xlabel+'='

	density_list = npzfile['density']
	tol_list     = npzfile['tol']
	N1_list      = npzfile['n1']
	
	for d,density in enumerate(density_list):
		for t,tol in enumerate(tol_list):
			M = [pow(n1,dim) for n1 in N1_list]
			fig, ax= plt.subplots(1,2,figsize=(15, 6))

			w = 0.1
			x = np.array(range(len(N1_list)))

			ax[0].bar(x, M/t_finufft[d,t,:],w, color="red",label='FINUFFT (28 threads)')
			ax[0].bar(x+w, M/t_cunfft[d,t,:],w, color="orange",label='cuNFFT')
			ax[0].bar(x+2*w, M/t_cufinufft[d,t,:],w, color="green",\
				label='cuFINUFFT, maxsubprob=32768')
			if nuptstype == 2:
				ax[0].bar(x+3*w, M/t_cufinufft_2[d,t,:],w, color="blue",\
					label='cuFINUFFT, maxsubprob=1024')
			formatter = ticker.ScalarFormatter()
			formatter.set_scientific(True)
			formatter.set_powerlimits((-1,1))

			ax[0].set_xlabel(xlabel)
			ax[0].set_title('Throughput (#NU pts/s)')
			ax[0].set_xticks(x+1.5*w)
			ax[0].set_xticklabels(N1_list)
			ax[0].yaxis.set_major_formatter(formatter)
			ax[0].legend(loc=0)
			ax[0].grid()

			ax[1].axhline(y=1, linestyle='--', color='k')
			ax[1].set_title('Speed up (s/s_FINUFFT)')
			ax[1].plot(x, t_finufft[d,t,:]/t_cunfft[d,t,:],'-o',color='orange',label='cuNFFT')
			ax[1].plot(x, t_finufft[d,t,:]/t_cufinufft[d,t,:],'-o',color='green',\
				label='cuFINUFFT, maxsubprob=32768')
			if nuptstype == 2:
				ax[1].plot(x, t_finufft[d,t,:]/t_cufinufft_2[d,t,:],'-o',color='blue',\
					label='cuFINUFFT, maxsubprob=1024')
			ax[1].set_xticks(x)
			ax[1].set_xticklabels(N1_list)
			ax[1].set_xlabel(xlabel)
			ax[1].set_ylim(bottom=0)
			ax[1].grid()
			ax[1].legend(loc=0)
			
			if nuptstype == 1:
				fig.suptitle('Uniform distributed pts ('+str(dim)+'D), Single Precision, tol='+str(tol)+'(ns ='+str(-np.log10(tol)+1)+'), density='+str(density), fontsize=15)
			if nuptstype == 2:
				fig.suptitle('Sphere quad pts ('+str(dim)+'D), Single Precision, tol='+str(tol)+'(ns ='+str(-np.log10(tol)+1)+'), density='+str(density), fontsize=15)
			fig.savefig('../plots/3d/density'+format(density, ".2e")+'_3d_t'+str(nuptstype)+'_+tol_'+str(tol)+'_'+date+'.png')
	plt.show()
if __name__== "__main__":
	main()
