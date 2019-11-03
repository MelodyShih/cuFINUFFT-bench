import numpy as np
import datetime

import matplotlib.pyplot as plt
from matplotlib import ticker

def main():
	d = datetime.datetime.today()
	date = d.strftime('%m%d')

	result_dir="../results/"
	t_cufinufft = np.load(result_dir+'workergpu14_cufinufft_spread_'+date+'.npy')
	t_finufft   = np.load(result_dir+'worker1160_finufft_spread_'+date+'.npy')
	t_cunfft    = np.load(result_dir+'workergpu14_cunfft_spread_'+date+'.npy')

	npzfile = np.load(result_dir+'param_'+date+'.npz')
	dim = 2
	density_list = npzfile['density']
	tol_list     = npzfile['tol']
	N1_list      = npzfile['n1']
	
	for d,density in enumerate(density_list):
		for t,tol in enumerate(tol_list):
			M = [pow(n1,dim) for n1 in N1_list]
			fig, ax= plt.subplots(1,2,figsize=(15, 6))

			w = 0.2
			x = np.array(range(len(N1_list)))

			ax[0].bar(x, M/t_finufft[d,t,:],w, color="darkviolet",label='FINUFFT (24 threads)')
			ax[0].bar(x+w, M/t_cunfft[d,t,:],w, color="violet",label='cuNFFT')
			ax[0].bar(x+2*w, M/t_cufinufft[d,t,:],w, color="deeppink",label='cuFINUFFT')
			formatter = ticker.ScalarFormatter()
			formatter.set_scientific(True)
			formatter.set_powerlimits((-1,1))

			ax[0].set_xlabel('nf1=nf2')
			ax[0].set_title('Throughput (#NU pts/s)')
			ax[0].set_xticks(x+1.5*w)
			ax[0].set_xticklabels(N1_list)
			ax[0].yaxis.set_major_formatter(formatter)
			ax[0].legend(loc=0)
			ax[0].grid()

			ax[1].axhline(y=1, linestyle='--', color='k')
			ax[1].set_title('Speed up (s/s_FINUFFT)')
			ax[1].plot(x, t_finufft[d,t,:]/t_cunfft[d,t,:],'-o',color='violet',label='cuNFFT')
			ax[1].plot(x, t_finufft[d,t,:]/t_cufinufft[d,t,:],'-o',color='deeppink',label='cuFINUFFT')
			ax[1].set_xticks(x)
			ax[1].set_xticklabels(N1_list)
			ax[1].set_xlabel('nf1=nf2')
			ax[1].set_ylim(bottom=0)
			ax[1].grid()
			ax[1].legend(loc=0)
			fig.suptitle('Uniform distributed pts, Single Precision, tol='+str(tol)+'(ns ='+str(-np.log10(tol)+1)+'), density='+str(density), fontsize=15)
			fig.savefig('../plots/density'+format(density, ".2e")+'_'+date+'.png')
	plt.show()
if __name__== "__main__":
	main()
